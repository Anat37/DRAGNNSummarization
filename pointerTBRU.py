from main import *

class LSTMState(ComponentLayerState):
        
    def reset(self):
        super().reset()
        self.states = []
        
    def add(self, token):
        token, state = token
        if self.is_solid:
            self.hiddens = token
            self.states.append(state)
        else:
            self.hiddens.append(token.squeeze(0))
            self.states.append(state)
        
    def get_last_state(self):
        return self.states[-1]
    
    def rearrange_last(self, ids):
        new_hiddens = torch.index_select(self.hiddens[-1], 0, ids)
        new_states = torch.index_select(self.states[-1], 1, ids)
        self.hiddens.pop(-1)
        self.hiddens.append(new_hiddens)
        self.states.pop(-1)
        self.states.append(new_states)
        
class PonterLSTMState(LSTMState):
        
    def reset(self):
        super().reset()
        self.distrib = []
        
    def add(self, token):
        token, state, distrib = token
        if self.is_solid:
            self.hiddens = token
            self.states.append(state)
            self.distrib = distrib
        else:
            self.hiddens.append(token.squeeze(0))
            self.states.append(state)
            self.distrib.append(distrib)
        
    def get_last_distrib(self):
        return self.distrib[-1]
    
    def rearrange_last(self, ids):
        super().rearrange_last(ids)
        new_hiddens = torch.index_select(self.distrib[-1], 0, ids)
        self.distrib.pop(-1)
        self.distrib.append(new_hiddens)

class AttentiveLSTMTBRU(nn.Module):
    def __init__(self, name, state_shape, is_solid, query_size, key_size, hidden_dim, input_hidden_layer, input_layer, is_first):
        super().__init__()
        
        self.is_solid = is_solid
        self.name = name
        self.state_shape = state_shape
        
        self._input_hidden_layer = input_hidden_layer
        self._rec = TaggerRecurrent(input_layer, name, is_first)
        
        self._query_layer = nn.Linear(query_size, hidden_dim)
        self._key_layer = nn.Linear(key_size, hidden_dim)
        self._energy_layer = nn.Linear(hidden_dim, 1)
        self._hidden_dim = hidden_dim
       
        self._rnn = nn.LSTM(query_size + key_size, hidden_dim)

    def forward(self, state, net):
        
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
            
        inputs = torch.cat((result, inputs), -1)
        
        if state == 0:
            if self._input_hidden_layer is None:
                hidden = None
            else:
                hidden = net.get_last(self._input_hidden_layer).unsqueeze(0)
                cstate = inputs.new_zeros((1, inputs.shape[1], self._hidden_dim))
                hidden = (hidden, cstate)
        else:
            hidden = net.get_last(self.name).unsqueeze(0)
            cstate = ((net.get_layer(self.name)).get_last_state())
            hidden = (hidden, cstate)

        #print(input_token.shape)
        output, (hidden, cstate) = self._rnn(inputs, hidden)
        
        if output is not None:
            net.add((output, cstate), self.name)
        return state, output
    
    def create_layer(self, net):
        comp_layer = LSTMState(self.name, self.is_solid)
        net.add_layer(comp_layer)        
        
class PointerTBRU(nn.Module):
    def __init__(self, name, state_shape, is_solid, query_size, key_size, hidden_dim, input_hidden_layer, input_layer, is_first):
        super().__init__()
        
        self.is_solid = is_solid
        self.name = name
        self.state_shape = state_shape
        
        self._input_hidden_layer = input_hidden_layer
        self._rec = TaggerRecurrent(input_layer, name, is_first)
        
        self._query_layer = nn.Linear(query_size, hidden_dim)
        self._key_layer = nn.Linear(key_size, hidden_dim)
        self._energy_layer = nn.Linear(hidden_dim, 1)
        self._hidden_dim = hidden_dim
       
        self._rnn = nn.LSTM(query_size + key_size, hidden_dim)

    def forward(self, state, net):
        
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
        attentions = []
        for i in range(key_value.size(0)):
            relevance = query_value + key_value[i]
            relevance = self._energy_layer(torch.tanh(relevance))
            f_att = F.softmax(relevance, 0)
            attentions.append(f_att)
            result.append((f_att * query).sum(0))
        if self.is_solid:
            result = torch.stack(result)
        else:
            result = result[0].unsqueeze(0)
            #print(inputs.shape)
            
        inputs = torch.cat((result, inputs), -1)
        
        if state == 0:
            if self._input_hidden_layer is None:
                hidden = None
            else:
                hidden = net.get_last(self._input_hidden_layer).unsqueeze(0)
                cstate = inputs.new_zeros((1, inputs.shape[1], self._hidden_dim))
                hidden = (hidden, cstate)
        else:
            hidden = net.get_last(self.name).unsqueeze(0)
            cstate = ((net.get_layer(self.name)).get_last_state())
            hidden = (hidden, cstate)

        #print(input_token.shape)
        output, (hidden, cstate) = self._rnn(inputs, hidden)
        
        if output is not None:
            net.add((output, cstate, None), self.name)
        return state, output
    
    def create_layer(self, net):
        comp_layer = PonterLSTMState(self.name, self.is_solid)
        net.add_layer(comp_layer)
        
class InputLayerWithBeamState(ComponentLayerState):
    def __init__(self, name, is_solid, inputs, beam_id):
        super().__init__(name, is_solid)
        self.hiddens = inputs
        self.beam_id = beam_id
        
    def add(self, token):
        token, beam_id = token
        if self.is_solid:
            self.hiddens = token
            self.beam_id = beam_id
        else:
            self.hiddens.append(token)
            self.beam_id.append(beam_id)
        
    def get_last_beam_id(self):
        return self.beam_id[-1]
    
    def rearrange_last(self, ids):
        new_hiddens = torch.index_select(self.hiddens[-1], 0, ids)
        new_states = torch.index_select(self.beam_id[-1], 1, ids)
        self.hiddens.pop(-1)
        self.hiddens.append(new_hiddens)
        self.beam_id.pop(-1)
        self.beam_id.append(new_states)
        
          
class BeamSearchProviderTBRU(nn.Module):
    def __init__(self, name, input_layer, working_layer, is_first, is_solid):
        super().__init__()
        
        self.is_solid = is_solid
        self.name = name
        self.working_layer = working_layer
        self._rec = TaggerRecurrent(input_layer, name, is_first)

    def forward(self, state, net):
        if self.is_solid or state == 0:
            return state, None
        input_layer = net.get_layer(self._rec._input_layer)
        work_layer = net.get_layer(self.working_layer)
        beam_id = input_layer.get_last_beam_id()
        work_layer.rearrange_last(beam_id)
        return state, None
    
    def create_layer(self, net):
        pass

        