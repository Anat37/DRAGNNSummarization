from main import *

class DRAGNNDecoderMaster(DRAGNNMaster):
    def __init__(self):
        super().__init__()
        
        self.net = NetState()
        self.encoder_list = []
        self.decoder_list = []
        self.decoder_input_layer = None
        self.decoder_state = 0
        
    def add_component_decoder(self, component):
        self.add_module(component.name, component)
        self.decoder_list.append(component)
        
    def add_component_encoder(self, component):
        self.add_module(component.name, component)
        self.encoder_list.append(component)
        
    def forward_encoder(self):
        for module in self.encoder_list:
            state, hidden = module(np.zeros(module.state_shape), self.net)
            while hidden is not None:
                state, hidden = module(state, self.net)
    
    def step_decoder(self):
        hidden = None
        for module in self.decoder_list:
            _, hidden = module(self.decoder_state, self.net)
        self.decoder_state = 1
        return hidden
    
    def forward(self, output_layer_name):
        self.forward_encoder()
        
        self.decoder_state = 0
        while True:
            hidden = self.step_decoder()
            if hidden is None:
                break
        
        return self.format_output(self.net.get_layer(output_layer_name).hiddens)
        
    def train_run(self, input_layer, target_layer):
        for module in self.decoder_list:
            module.is_solid = True
            if module._rec._is_first:
                module._rec._input_layer = target_layer.name
        
        self.build_net(input_layer)
        self.net.add_layer(target_layer)   
        return self.forward(self.decoder_list[-1].name)
    
    def eval_run_encoder(self, input_layer): #TODO
        self.decoder_input_layer = InputLayerState("decoder_input_layer", False, [])
        
        for module in self.decoder_list:
            module.is_solid = False
            if module._rec._is_first:
                module._rec._input_layer = self.decoder_input_layer.name
        
        self.build_net(input_layer)
        self.net.add_layer(self.decoder_input_layer)
        self.decoder_state = 0
        return self.forward_encoder()
    
    def decode(self, symbol):
        self.decoder_input_layer.add(symbol)
        return self.step_decoder()
    
class LSTMEncoderComputer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self._hidden_size = hidden_size
        self._rnn = nn.LSTM(input_size, hidden_size)
        self._state = None

    def forward(self, state, input_token):
        inputs, hidden = input_token
        if inputs is None:
            return state, None
        if not isinstance(hidden, tuple):
            if state == 0:
                self._state = inputs.new_zeros((1, inputs.shape[1], self._hidden_size))
            if hidden is not None:
                hidden = (hidden.unsqueeze(0), self._state)
                
        #print(input_token.shape)
        output, (hidden, self._state) = self._rnn(inputs, hidden)

        return state, output
    
class LSTMEncoderRecurrent():
    def __init__(self, input_hidden_layer, input_layer, self_name, is_first):
        super().__init__()
        
        self._self_name = self_name
        self._input_hidden_layer = input_hidden_layer
        self._input_layer = input_layer
        self._is_first = is_first

    def get(self, state, net, is_solid):
        if state == 0:
            if self._input_hidden_layer is None:
                hidden = None
            else:
                hidden = net.get_last(self._input_hidden_layer)
        else:
            hidden = net.get_last(self._self_name)
        
        if is_solid:
            inputs = net.get_full(self._input_layer, self._self_name)
        else:
            inputs = net.get_last(self._input_layer)
        return (inputs, hidden)
    
    
class AdditiveAttentiveLSTMEncoderRecurrent(nn.Module):
    def __init__(self, query_size, key_size, hidden_dim, input_hidden_layer, input_layer, self_name, is_first):
        super().__init__()
        
        self._self_name = self_name
        self._input_hidden_layer = input_hidden_layer
        self._input_layer = input_layer
        self._is_first = is_first
        
        self._query_layer = nn.Linear(query_size, hidden_dim)
        self._key_layer = nn.Linear(key_size, hidden_dim)
        self._energy_layer = nn.Linear(hidden_dim, 1)
        self._hidden_dim = hidden_dim

    def get(self, state, net, is_solid):
        if state == 0:
            if self._input_hidden_layer is None:
                hidden = None
            else:
                hidden = net.get_last(self._input_hidden_layer)
        else:
            hidden = net.get_last(self._self_name)
        
        if is_solid:
            inputs = net.get_full(self._input_layer, self._self_name)
        else:
            inputs = net.get_last(self._input_layer)
        
        if inputs is None:
            return (None, None)
        query = net.get_all(self._input_hidden_layer)
        query_value = self._query_layer(query)
        key_value = self._key_layer(inputs)
        result = []
        for i in range(key_value.size(0)):
            relevance = query_value + key_value[i]
            relevance = self._energy_layer(torch.tanh(relevance))
            f_att = F.softmax(relevance, 0)
            result.append((f_att * query).sum(0))
        if is_solid:
            result = torch.stack(result)
        else:
            result = result[0].unsqueeze(0)
        return (torch.cat((result, inputs), -1), hidden)