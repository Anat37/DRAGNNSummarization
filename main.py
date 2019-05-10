import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.cuda import FloatTensor, LongTensor

class NetState():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.components = []
    
    def add_layer(self, component):
        self.components.append(component)
        
    def has_layer(self, layer_name):
        for c in self.components:
            if c.name == layer_name:
                return True
        return False
    
    def get_layer(self, layer_name):
        for c in self.components:
            if c.name == layer_name:
                return c
        return None
            
    def get_outputs(self):
        return self.components[-1].hiddens
        
    def get_value_by_name(self, name, index, module_name):
        for c in self.components:
            if c.name == name:
                return c.get(index, module_name)
        return None
            
    def get_full(self, name, module_name):
        for c in self.components:
            if c.name == name:
                return c.get_full(module_name)
        return None
    
    def get_all(self, name):
        for c in self.components:
            if c.name == name:
                return c.get_all()
        return None
    
    def get_last(self, name):
        for c in self.components:
            if c.name == name:
                return c.get_last()
        return None
        
    def add(self, hidden, name):
        for c in self.components:
            if c.name == name:
                c.add(hidden)
                return
        
class ComponentLayerState():
    def __init__(self, name, is_solid):
        self.name = name
        self.is_solid = is_solid
        self.reset()
        
    def reset(self):
        self.pos = {}
        self.hiddens = []
    
    def get(self, index, module_name):
        if not module_name in self.pos:
            self.pos[module_name] = -1
        if index > 0: 
            if self.pos[module_name] + 1 >= len(self.hiddens):
                return None
            else:
                self.pos[module_name] += 1
                return self.hiddens[self.pos[module_name]]
    
    def get_full(self, module_name):
        if not module_name in self.pos:
            self.pos[module_name] = -1
            
        if self.pos[module_name] < len(self.hiddens) - 1:
            result = self.hiddens[self.pos[module_name] + 1: len(self.hiddens)]
            self.pos[module_name] = len(self.hiddens) - 1
            return result
        else:
            return None
    
    def get_all(self):
        return self.hiddens
    
    def add(self, token):
        if self.is_solid:
            self.hiddens = token
        else:
            self.hiddens.append(token)
            
    def get_last(self):
        return self.hiddens[-1]
    
    def rearrange_last(self, ids, dim = 0):
        new_hiddens = torch.index_select(self.hiddens[-1], dim, ids)
        self.hiddens.pop(-1)
        self.hiddens.append(new_hiddens)
            
class InputLayerState(ComponentLayerState):
    def __init__(self, name, is_solid, inputs):
        super().__init__(name, is_solid)
        self.hiddens = inputs
        
class RNNComputer(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        
        self._hidden_size = hidden_size
        self._hidden = nn.Linear(hidden_size + input_size, hidden_size)

    def forward(self, state, input_token):
        inputs, hidden = input_token
        if inputs is None:
            return state, None
        if hidden is None:
            hidden = inputs.new_zeros(inputs.size(0), self._hidden_size)
        x = torch.cat((hidden, inputs), -1)
        hidden = torch.tanh(self._hidden(x))

        return state, hidden
    
class RNNSolidComputer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self._hidden_size = hidden_size
        self._rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)

    def forward(self, state, input_token):
        if input_token is None:
            return state, None
        #if hidden is None:
        hidden = (input_token.new_zeros((2, input_token.shape[1], self._hidden_size)),
                  input_token.new_zeros((2, input_token.shape[1], self._hidden_size)))
        #print(input_token.shape)
        output, hidden = self._rnn(input_token, hidden)

        return state, output
    
class TaggerComputer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self._hidden_size = hidden_size
        self._hidden = nn.Linear(input_size, hidden_size)

    def forward(self, state, input_token):
        if input_token is None:
            return state, None
        hidden = self._hidden(input_token)
        return state, hidden
    
class EmbeddingComputer(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_idx):
        super().__init__()
        
        self._embed = nn.Embedding(vocab_size, hidden_size, padding_idx = pad_idx)
        self._vocab_size = vocab_size

    def forward(self, state, input_token):
        if input_token is None:
            return state, None
        #print(input_token.shape)
        hidden = self._embed(input_token)
        return state, hidden
    
class RNNRecurrent():
    def __init__(self, input_name, self_name):
        super().__init__()
        
        self._input_name = input_name
        self._self_name = self_name

    def get(self, state, net, is_solid):
        inputs = net.get_value_by_name(self._input_name, 1, self._self_name)
        hidden = net.get_value_by_name(self._self_name, 1, self._self_name)
        return inputs, hidden
    
class RNNSolidRecurrent():
    def __init__(self, input_name, self_name):
        super().__init__()
        
        self._input_name = input_name
        self._self_name = self_name

    def get(self, state, net, is_solid):
        inputs = net.get_full(self._input_name, self._self_name)
        return inputs
    
class TaggerRecurrent():
    def __init__(self, input_name, self_name, is_first):
        super().__init__()
        
        self._input_layer = input_name
        self._self_name = self_name
        self._is_first = is_first

    def get(self, state, net, is_solid):
        if is_solid:
            inputs = net.get_full(self._input_layer, self._self_name)
            if isinstance(inputs, list):
                inputs = torch.stack(inputs)
                inputs.requires_grad_()
        else:
            inputs = net.get_value_by_name(self._input_layer, 1, self._self_name)   
        return inputs

class AbstractTBRU(nn.Module):
    def __init__(self, name, state_shape, is_solid, solid_modifiable = True):
        super().__init__()
        
        self._is_solid = is_solid
        self.state_shape = state_shape
        self.name = name
        self._solid_modifiable = solid_modifiable
    
    def create_layer(self, net):
        comp_layer = ComponentLayerState(self.name, self._is_solid)
        net.add_layer(comp_layer)
        
    def set_solid(self, is_solid):
        if self._solid_modifiable:
            self._is_solid = is_solid
            
    def set_beam_search(self, flag):
        pass
    
class TBRU(AbstractTBRU):
    def __init__(self, name, recurrent, computer, state_shape, is_solid, solid_modifiable=True):
        super().__init__(name, state_shape, is_solid, solid_modifiable)
        
        self._rec = recurrent
        self._comp = computer

    def forward(self, state, net):
        #print(self.name)
        state, hidden = self._comp(state, (self._rec.get(state, net, self._is_solid)))
  
        if hidden is not None:
            net.add(hidden, self.name)
        return state, hidden

    
class DRAGNNMaster(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = NetState()
        
    def add_component(self, component):
        self.add_module(component.name, component)

    def prepare_net(self, net):
        for c in self._modules:
            self._modules[c].create_layer(net)
        return net
    
    def build_net(self, input_layer):
        if self.net is not None:
            del self.net
            self.net = NetState()
        self.net.reset()
        self.net.add_layer(input_layer)
        self.prepare_net(self.net)
    
    def format_output(self, output):
        if not isinstance(output, list):
            return output
        else:
            output = torch.stack(output)
            output.requires_grad_()
            return output
    
    def forward(self, input_layer):
        self.build_net(input_layer)
        for c in self._modules:
            module = self._modules[c]
            state, hidden = module(np.zeros(module.state_shape), self.net)
            while hidden is not None:
                state, hidden = module(state, self.net)
                
        return self.format_output(self.net.components[-1].hiddens)
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        
    def save_checkpoint(self, epoch, optimizer, filename='checkpoint.pth.tar'):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, filename)    
