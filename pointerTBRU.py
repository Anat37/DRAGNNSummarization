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
            self.hiddens.append(token)
            #print(self.hiddens[-1].shape)
            self.states.append(state)
        
    def get_last_state(self):
        return self.states[-1]
    
    def rearrange_last(self, ids, dim = 0):
        new_hiddens = torch.index_select(self.hiddens[-1], 0, ids)
        #new_states = torch.index_select(self.states[-1], 1, ids)
        self.hiddens.pop(-1)
        self.hiddens.append(new_hiddens)
        #self.states.pop(-1)
        #self.states.append(new_states)
        
class PonterLSTMState(LSTMState):
        
    def reset(self):
        super().reset()
        self.distrib = []
        self.pgen = []
        
    def add(self, token):
        token, state, distrib, pgen = token
        if self.is_solid:
            self.hiddens = token
            self.states.append(state)
            self.distrib = distrib
            self.pgen = pgen
        else:
            self.hiddens.append(token.squeeze(0))
            self.states.append(state)
            self.distrib.append(distrib)
            self.pgen.append(pgen)
        
    def get_last_distrib(self):
        return self.distrib[-1]
    
    def get_last_pgen(self):
        return self.pgen[-1]
    
    def get_full_distrib(self):
        return self.distrib
    
    def get_full_pgen(self):
        return self.pgen
    
    def rearrange_last(self, ids, dim=0):
        super().rearrange_last(ids)
        new_hiddens = torch.index_select(self.distrib[-1], 0, ids)
        self.distrib.pop(-1)
        self.distrib.append(new_hiddens)
        new_hiddens = torch.index_select(self.pgen[-1], 0, ids)
        self.pgen.pop(-1)
        self.pgen.append(new_hiddens)
        
class HorizontalRecurrent():
    def __init__(self, input_name, self_name, is_first):
        super().__init__()
        
        self._input_layer = input_name
        self._self_name = self_name
        self._is_first = is_first
    
    def get_simple(self, state, net, is_solid):
        if is_solid:
            inputs = net.get_full(self._input_layer, self._self_name)
            if isinstance(inputs, list):
                inputs = torch.stack(inputs)
                inputs.requires_grad_()
        else:
            inputs = net.get_value_by_name(self._input_layer, 1, self._self_name)
        return inputs
        
    def get(self, state, net, is_solid, batch_size, hidden_size):
        input_layer = net.get_layer(self._input_layer)
        #print(input_layer.hiddens)
        inputs = self.get_simple(state, net, is_solid)
        if state == 0 and inputs is None:
            inputs = torch.cuda.FloatTensor(batch_size, hidden_size).fill_(0)
            input_layer.add(inputs)
            inputs = self.get_simple(state, net, is_solid)
        
        #print(inputs)
        return inputs
        
class ConcatTBRU(AbstractTBRU):
    def __init__(self, name, is_solid, input_layer, second_layer, is_first, solid_modifiable=True):
        super().__init__(name, (1,), is_solid, solid_modifiable)
        
        self._rec = TaggerRecurrent(input_layer, name, is_first)
        self._rec2 = TaggerRecurrent(second_layer, name, is_first)

    def forward(self, state, net):
        inputs1 = self._rec.get(state, net, self._is_solid)
        inputs2 = self._rec2.get(state, net, self._is_solid)

        if inputs1 is None or inputs2 is None:
            return (state, None)
        
        #print(inputs1.shape)
        #print(inputs2.shape)
        inputs = torch.cat((inputs1, inputs2), -1)
     
        if inputs is not None:
            net.add(inputs, self.name)
        return state, inputs
    
class ConcatStateTBRU(AbstractTBRU):
    def __init__(self, name, is_solid, input_layer, second_layer, hidden_dim, is_first, solid_modifiable=True):
        super().__init__(name, (1,), is_solid, solid_modifiable)
        
        self._rec = TaggerRecurrent(input_layer, name, is_first)
        self._rec2 = HorizontalRecurrent(second_layer, name, False)
        self._hidden_dim = hidden_dim

    def forward(self, state, net):
        inputs1 = self._rec.get(state, net, self._is_solid)
        
        if inputs1 is None:
            return (state, None)
        
        if not self._is_solid:
            if inputs1.dim() < 3:
                inputs1 = inputs1.unsqueeze(0)
        inputs2 = self._rec2.get(state, net, self._is_solid, inputs1.shape[1], self._hidden_dim).unsqueeze(0)
        
        #print(inputs1.shape)
        #print(inputs2.shape)
        inputs = torch.cat((inputs1, inputs2), -1)
     
        if inputs is not None:
            net.add(inputs, self.name)
        return state, inputs
        
class AttentionTBRU(AbstractTBRU):
    def __init__(self, name, is_solid, query_size, key_size, hidden_dim, input_layer, query_layer, is_first, solid_modifiable=True):
        super().__init__(name, (1,), is_solid, solid_modifiable)
        
        self._rec = TaggerRecurrent(input_layer, name, is_first)
        
        self._query_linear = nn.Linear(query_size, hidden_dim)
        self._key_layer = nn.Linear(key_size, hidden_dim)
        self._energy_linear = nn.Linear(hidden_dim, 1)
        self._hidden_dim = hidden_dim
        self._query_layer = query_layer
        self._query_value = None

    def forward(self, state, net):
        
        if self._is_solid:
            inputs = net.get_full(self._rec._input_layer, self.name)
        else:
            inputs = net.get_value_by_name(self._rec._input_layer, 1, self.name)
            
        if inputs is None:
            return (state, None)
        
        if not self._is_solid:
            inputs = inputs.unsqueeze(0)
        
        if state == 0:
            query = net.get_all(self._query_layer)
            self._query_value = self._query_linear(query)
            
        key_value = self._key_layer(inputs)
        attentions = []
        for i in range(key_value.size(0)):
            relevance = self._query_value + key_value[i]
            relevance = self._energy_linear(torch.tanh(relevance))
            f_att = torch.softmax(relevance, 0)
            attentions.append(f_att.squeeze(-1))
        if self._is_solid:
            attentions = torch.stack(attentions)
        else:
            attentions = attentions[0]
        
        if attentions is not None:
            net.add(attentions, self.name)
        return state, attentions
    
class CoverageAttentionTBRU(AbstractTBRU):
    def __init__(self, name, coverage_layer, is_solid, query_size, key_size, hidden_dim, input_layer, query_layer, is_first, solid_modifiable=True):
        super().__init__(name, (1,), is_solid, solid_modifiable)
        
        self._rec = TaggerRecurrent(input_layer, name, is_first)
        
        self._coverage_layer = coverage_layer
        self._query_linear = nn.Linear(query_size, hidden_dim)
        self._key_layer = nn.Linear(key_size, hidden_dim)
        self._energy_linear = nn.Linear(hidden_dim, 1)
        self._coverage_linear = nn.Linear(1, hidden_dim)
        self._hidden_dim = hidden_dim
        self._query_layer = query_layer
        self._query_value = None
        self._coverage = HorizontalRecurrent(coverage_layer, name, False)

    def forward(self, state, net):
        
        query = net.get_all(self._query_layer)
        if self._is_solid:
            #coverage = query.new_zeros((query.shape[0], query.shape[1], 1))
            inputs = net.get_full(self._rec._input_layer, self.name)
        else:
            #coverage = net.get_value_by_name(self._coverage_layer, 1, self.name)
            #if coverage is None:
                #coverage = query.new_zeros((query.shape[0], query.shape[1], 1))
                #net.add(coverage, self._coverage_layer)
            inputs = net.get_value_by_name(self._rec._input_layer, 1, self.name)
        
        if inputs is None:
            return (state, None)
        
        coverage = (self._coverage.get(state, net, self._is_solid, query.shape[0], query.shape[1])).unsqueeze(-1)
        
        if not self._is_solid:
            inputs = inputs.unsqueeze(0)
        
        if state == 0:
            self._query_value = self._query_linear(query)
            
        key_value = self._key_layer(inputs)
        attentions = []
        for i in range(key_value.size(0)):
            relevance = self._query_value + key_value[i]
            coverage_feature = self._coverage_linear(coverage)
            #print(relevance.shape)
            #print(coverage_feature.shape)
            
            relevance = self._energy_linear(torch.tanh(relevance + coverage_feature))            
            f_att = torch.softmax(relevance, 0)
            coverage = coverage + f_att
            net.add(coverage.squeeze(-1), self._coverage_layer)         
            attentions.append(f_att.squeeze(-1))
        if self._is_solid:
            attentions = torch.stack(attentions)
        else:
            attentions = attentions[0]
        
        if attentions is not None:
            net.add(attentions, self.name)
        return state, attentions
    
    def create_layer(self, net):
        comp_layer = ComponentLayerState(self.name, self._is_solid)
        net.add_layer(comp_layer)
        comp_layer = ComponentLayerState(self._coverage_layer, False)
        net.add_layer(comp_layer)
    
class ContextTBRU(AbstractTBRU):
    def __init__(self, name, is_solid, attention_layer, query_layer, is_first, solid_modifiable=True):
        super().__init__(name, (1,), is_solid, solid_modifiable)
        
        self._rec = TaggerRecurrent(attention_layer, name, is_first)
        self._query_layer = query_layer

    def forward(self, state, net): 
        attention = self._rec.get(state, net, self._is_solid)
        query = net.get_all(self._query_layer)
        if attention is None or query is None:
            return (state, None)
        
        attention = attention.unsqueeze(-1)
        if self._is_solid:
            result = []
            for i in range(attention.shape[0]):
                result.append((attention[i] * query).sum(0))
            result = torch.stack(result)
        else:
            result = (attention * query).sum(0)
        
        if result is not None:
            net.add(result, self.name)
        return state, result

class LSTMTBRU(AbstractTBRU):
    def __init__(self, name, state_name, is_solid, input_size, hidden_dim, input_hidden_layer, input_layer, is_first, solid_modifiable=True):
        super().__init__(name, (1,), is_solid, solid_modifiable)
        
        self._input_hidden_layer = input_hidden_layer
        self._rec = TaggerRecurrent(input_layer, name, is_first)
        self._state_name = state_name
        self._hidden_dim = hidden_dim
        self._cstate = HorizontalRecurrent(state_name, name, False)
        self._rnn = nn.LSTM(input_size, hidden_dim)

    def forward(self, state, net):
        inputs = self._rec.get(state, net, self._is_solid)
        if inputs is None:
            return (state, None)
        
        if not self._is_solid:
            if inputs.dim() < 3:
                inputs = inputs.unsqueeze(0)
        
        #print(inputs.shape)
        cstate = self._cstate.get(state, net, self._is_solid, inputs.shape[1], self._hidden_dim)
        cstate = cstate.unsqueeze(0)
        if state == 0:
            if self._input_hidden_layer is None:
                hidden = None
            else:
                hidden = net.get_last(self._input_hidden_layer).unsqueeze(0)
                #cstate = inputs.new_zeros((1, inputs.shape[1], self._hidden_dim))
        else:
            hidden = net.get_last(self.name).unsqueeze(0)
            #cstate = ((net.get_layer(self._state_name)).get_last()).unsqueeze(0)
        hidden = (hidden, cstate)  
            
        #print(input_token.shape)
        output, (hidden, cstate) = self._rnn(inputs, hidden)
        
        if not self._is_solid:
            output = output.squeeze(0)
            
        if output is not None:
            net.add(output, self.name)
            net.add(cstate.squeeze(0), self._state_name)
        return state, output
    
    def create_layer(self, net):
        comp_layer = ComponentLayerState(self.name, self._is_solid)
        net.add_layer(comp_layer)
        comp_layer = ComponentLayerState(self._state_name, False)
        net.add_layer(comp_layer)
               
class PointerTBRU(AbstractTBRU):
    def __init__(self, name, is_solid, input_size, vocab_size, input_attention_layer, input_context_layer, input_distibution_layer, input_decoder_layer, input_encoder_layer, input_lstm_layer, lstm_state_layer, solid_modifiable=True):
        super().__init__(name, (1,), is_solid, solid_modifiable)
        
        self._input_encoder_layer = input_encoder_layer
        self._input_lstm_layer = input_lstm_layer
        self._input_distibution_layer = input_distibution_layer
        self._input_context_layer = input_context_layer
        self._input_decoder_layer = input_decoder_layer
        self._rec = TaggerRecurrent(input_attention_layer, name, False)
        self._cstate =  HorizontalRecurrent(lstm_state_layer, name, False)
        self._vocab_size = vocab_size
        self._linear = nn.Linear(input_size, 1)

    def forward(self, state, net):
        attention = self._rec.get(state, net, self._is_solid)
        words = net.get_all(self._input_encoder_layer)
        
        if self._is_solid:
            lstm_hiddens = net.get_full(self._input_lstm_layer, self.name)
            context = net.get_full(self._input_context_layer, self.name)
            distribs = net.get_full(self._input_distibution_layer, self.name)
            lstm_input = net.get_full(self._input_decoder_layer, self.name)
        else:
            lstm_hiddens = net.get_value_by_name(self._input_lstm_layer, 1, self.name)
            context = net.get_value_by_name(self._input_context_layer, 1, self.name)
            distribs = net.get_value_by_name(self._input_distibution_layer, 1, self.name)
            lstm_input = net.get_value_by_name(self._input_decoder_layer, 1, self.name)
        
        if words is None or attention is None or lstm_hiddens is None or context is None or distribs is None or lstm_input is None:
            return state, None
        
        cstate = self._cstate.get(state, net, self._is_solid, lstm_input.shape[1], lstm_hiddens.shape[-1])
        #cstate = cstate.unsqueeze(0)
        
        if not self._is_solid:
            lstm_input = lstm_input.squeeze(0)
        
        attention = attention.transpose(-2, -1)
        words = words.transpose(-2, -1)
        
        if self._is_solid:
            batch_size = distribs.shape[1]
        else:
            batch_size = distribs.shape[0]
        
        pgen = self._linear(torch.cat((context, lstm_hiddens, lstm_input, cstate), -1))
        #print(pgen)
        eps = 1e-5
        eps = torch.ones_like(pgen) * eps
        pgen = torch.sigmoid(torch.max(pgen,eps))
        
        if distribs.shape[-1] < self._vocab_size:
            if self._is_solid:
                zeros = distribs.new_zeros((distribs.shape[0], batch_size, self._vocab_size - distribs.shape[2]))
            else:
                zeros = distribs.new_zeros((batch_size, self._vocab_size - distribs.shape[1]))
            distribs = torch.cat((distribs, zeros), -1)
        
        
        vocab_dist = (1 - pgen) * torch.softmax(distribs, -1)
        #vocab_dist = pgen * distribs
        #print(attention.shape)
        #print(pgen.shape)
        #print(words.shape)
        
        attn_dist = pgen * attention
        
        if self._is_solid:
            for i in range(attn_dist.shape[0]):
                vocab_dist[i] = vocab_dist[i].scatter_add(1, words, attn_dist[i])
        else:
            vocab_dist = vocab_dist.scatter_add(1, words, attn_dist)
        
        if vocab_dist is not None:
            net.add(vocab_dist, self.name)
        return state, vocab_dist
        
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
    
    def rearrange_last(self, ids, dim=0):
        new_hiddens = torch.index_select(self.hiddens[-1], 0, ids)
        new_states = torch.index_select(self.beam_id[-1], 1, ids)
        self.hiddens.pop(-1)
        self.hiddens.append(new_hiddens)
        self.beam_id.pop(-1)
        self.beam_id.append(new_states)
        
class BeamSearchProviderTBRU(AbstractTBRU):
    def __init__(self, name, input_layer, dim, working_layer, is_first, is_solid, solid_modifiable=True):
        super().__init__(name, (1,), is_solid, solid_modifiable)
        
        self.working_layer = working_layer
        self._dim = dim
        self._active = False
        self._rec = TaggerRecurrent(input_layer, name, is_first)

    def forward(self, state, net):
        if (not self._active) or state == 0:
            return state, None
        input_layer = net.get_layer(self._rec._input_layer)
        work_layer = net.get_layer(self.working_layer)
        beam_id = input_layer.get_last_beam_id()
        work_layer.rearrange_last(beam_id, self._dim)
        return state, None
    
    def create_layer(self, net):
        pass
    
    def set_beam_search(self, flag):
        self._active = flag

        