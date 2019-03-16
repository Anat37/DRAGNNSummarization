from main import *

class DRAGNNDecoderMaster(DRAGNNMaster):
    def __init__(self):
        super().__init__()
        
        self.net = NetState()
        self.encoder_list = []
        self.decoder_list = []
        
    def add_component_decoder(self, component):
        self.add_module(component.name, component)
        self.decoder_list.append(component)
        
    def add_component_encoder(self, component):
        self.add_module(component.name, component)
        self.encoder_list.append(component)
        
    def forward(self, input_layer, output_layer_name):
        for module in self.encoder_list:
            state, hidden = module(np.zeros(module.state_shape), self.net)
            while hidden is not None:
                state, hidden = module(state, self.net)
        
        state = 0
        while True:
            for module in self.decoder_list:
                _, hidden = module(state, self.net)
            state = 1
            if hidden is None:
                break
        
        output = self.net.get_layer(output_layer_name).hiddens
        if not isinstance(output, list):
            return output
        else:
            output = torch.stack(output)
            output.requires_grad_()
            return output
        
    def train_run(self, input_layer, target_layer):
        for module in self.decoder_list:
            module.is_solid = True
            if module._rec._is_first:
                module._rec._input_layer = target_layer.name
        
        self.build_net(input_layer)
        self.net.add_layer(target_layer)   
        return self.forward(input_layer, self.decoder_list[-1].name)
    
    def eval_run(self, input_layer, start_symbol): #TODO
        for module in self.decoder_list:
            module.is_solid = False
            if module._rec._is_first:
                module._rec._input_layer = input_layer.name
        
        self.build_net(input_layer)
        return self.forward(input_layer, self.decoder_list[-1].name)
    
        
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
