from main import *

class PointerTBRU(nn.Module):
    def __init__(self, name, state_shape, is_solid, query_size, key_size, hidden_dim, input_hidden_layer, input_layer, is_first):
        super().__init__()
        
        self.is_solid = is_solid
        self.name = name
        self.state_shape = state_shape
        
        self._input_hidden_layer = input_hidden_layer
        self._rec = TaggerRecurrent(input_layer, name, is_first)
        self._rec._input_layer = input_layer
        self._rec._is_first = is_first
        
        self._query_layer = nn.Linear(query_size, hidden_dim)
        self._key_layer = nn.Linear(key_size, hidden_dim)
        self._energy_layer = nn.Linear(hidden_dim, 1)
        self._hidden_dim = hidden_dim
       
        self._rnn = nn.LSTM(hidden_dim + key_size, hidden_dim)
        self._state = None

    def forward(self, state, net):
        #state, hidden = self._comp(state, (self._rec.get(state, net, self.is_solid)))
        
        if state == 0:
            if self._input_hidden_layer is None:
                hidden = None
            else:
                hidden = net.get_last(self._input_hidden_layer)
        else:
            hidden = net.get_last(self.name)
            if not self.is_solid:
                hidden = hidden.squeeze(0)
        
        if self.is_solid:
            inputs = net.get_full(self._rec._input_layer, self.name)
        else:
            inputs = net.get_last(self._rec._input_layer)
        
        if inputs is None:
            return (state, None)
        query = net.get_all(self._input_hidden_layer)
        query_value = self._query_layer(query)
        key_value = self._key_layer(inputs)
        result = []
        for i in range(key_value.size(0)):
            relevance = query_value + key_value[i]
            relevance = self._energy_layer(torch.tanh(relevance))
            f_att = F.softmax(relevance, 0)
            result.append((f_att * query).sum(0))
        if self.is_solid:
            result = torch.stack(result)
        else:
            result = result[0].unsqueeze(0)
            #print(inputs.shape)
            
        inputs, hidden = torch.cat((result, inputs), -1), hidden
        
        if not isinstance(hidden, tuple):
            if state == 0:
                self._state = inputs.new_zeros((1, inputs.shape[1], self._hidden_dim))
            if hidden is not None:
                hidden = (hidden.unsqueeze(0), self._state)
                
        #print(input_token.shape)
        output, (hidden, self._state) = self._rnn(inputs, hidden)
        
        if hidden is not None:
            net.add(hidden, self.name)
        return state, hidden
    
    def create_layer(self, net):
        comp_layer = ComponentLayerState(self.name, self.is_solid)
        net.add_layer(comp_layer)
        