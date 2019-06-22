from decoder import *
from data import *
from pointerTBRU import BeamSearchProviderTBRU, BiLSTMTBRU, AttentionTBRU, ConcatTBRU, PointerTBRU, ContextTBRU, CoverageAttentionTBRU, ConcatStateTBRU, UnknownVocabComputer

def build_decoder_model(mask_layer=None):
    hidden_dim = 256
    emb_dim= 128 
    
    master = DRAGNNDecoderMaster()
    
    embeddings_computer = EmbeddingComputer(VOCAB_SIZE, emb_dim, PAD_TOKEN_ID)
    unk_comp = UnknownVocabComputer(VOCAB_SIZE, UNKNOWN_TOKEN_ID)
    #master.add_component_encoder(TBRU("embed_input", TaggerRecurrent("input", "embed_input", False), unk_comp, (1,), True).cuda())
    master.add_component_encoder(TBRU("embed", TaggerRecurrent("input", "embed", False), embeddings_computer, (1,), True).cuda())
    #master.add_component(TBRU("extractive", TaggerRecurrent("embed", "extractive"), TaggerComputer(1000, 1000), (1,), True).cuda())
    #master.add_component_encoder(TBRU("bilstm", RNNSolidRecurrent("embed", "bilstm"), BILSTMSolidComputer(emb_dim, hidden_dim), (1,), True).cuda())
    
    encoder_BiLSTM = BiLSTMTBRU("bilstm", "bilstm_state_layer", emb_dim, hidden_dim, "embed", is_solid = True, bidirectional=True, solid_modifiable=False)
    master.add_component_encoder(encoder_BiLSTM.cuda())
    
    master.add_component_encoder(TBRU("bilstm_reduced", TaggerRecurrent("bilstm", "bilstm_reduced", False), TaggerComputer(hidden_dim * 2, hidden_dim), (1,), is_solid=True).cuda())
    #master.add_component_encoder(TBRU("bilstm_state_reduced", TaggerRecurrent("bilstm_state_layer", "bilstm_state_reduced", False), TaggerComputer(hidden_dim * 2, hidden_dim), (1,), is_solid=True).cuda())
    
    
    #master.add_component_decoder(TBRU("decoder_embed_input", TaggerRecurrent(None, "decoder_embed_input", True), unk_comp, (1,), True).cuda())
    master.add_component_decoder(TBRU("decoder_embed", TaggerRecurrent(None, "decoder_embed", True),embeddings_computer, (1,), is_solid=True).cuda())
    
    master.add_component_decoder(BeamSearchProviderTBRU("beamProvider", None, 0, "decoder", True, is_solid=False))
    master.add_component_decoder(BeamSearchProviderTBRU("beamProvider1", None, 0, "lstm_state_layer", True, is_solid=False))
    master.add_component_decoder(BeamSearchProviderTBRU("beamProvider2", None, 1, "coverage_layer", True, is_solid=False))
    master.add_component_decoder(BeamSearchProviderTBRU("beamProvider3", None, 0, "context", True, is_solid=False))
    
    master.add_component_decoder(ConcatStateTBRU("decoder_input_concat", False, "decoder_embed", 'context', hidden_dim, False, solid_modifiable=False))
    
    master.add_component_decoder(TBRU("decoder_input", TaggerRecurrent("decoder_input_concat", "decoder_input", False), TaggerComputer(emb_dim + hidden_dim, hidden_dim), (1,), is_solid=False, solid_modifiable=False).cuda())
    pTBRU = BiLSTMTBRU("decoder", "lstm_state_layer", hidden_dim, hidden_dim, "decoder_input", input_hidden_layer="bilstm_reduced",  solid_modifiable=False)
#input_state_layer = "bilstm_state_reduced",    
    master.add_component_decoder(pTBRU.cuda())
    
    master.add_component_decoder(ConcatStateTBRU('attention_input', False, 'decoder', 'lstm_state_layer', hidden_dim, is_first=False, solid_modifiable=False))
    
    master.add_component_decoder(CoverageAttentionTBRU("attention_layer", "coverage_layer", False, 2*hidden_dim, 2*hidden_dim, hidden_dim, 'attention_input','bilstm', is_first=False, mask_layer=mask_layer,solid_modifiable=False).cuda())
    master.add_component_decoder(ContextTBRU('context', False, 'attention_layer', 'bilstm_reduced', is_first=False, solid_modifiable=False))
    master.add_component_decoder(ConcatTBRU('context_concat', False, 'decoder', 'context', is_first=False, solid_modifiable=False))
    master.add_component_decoder(TBRU("output", TaggerRecurrent("context_concat", "output", False), TaggerComputer(2 * hidden_dim, VOCAB_SIZE), (1,), is_solid=False, solid_modifiable=False).cuda())
    master.add_component_decoder(PointerTBRU("pointer_final", False, 4*hidden_dim, VOCAB_SIZE + ADDITIONAL_WORDS, "attention_layer", "context", "output", "decoder_input", "input_full", "decoder","lstm_state_layer", solid_modifiable=False).cuda())
    
    normal_std = 1e-4
    uniform_mag = 0.02
    
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=normal_std)
            torch.nn.init.normal_(m.bias.data, std=normal_std)
        if type(m) == nn.LSTM:
            for name, param in m.named_parameters():
                if 'bias' in name:
                    torch.nn.init.normal_(param, std=normal_std)
                elif 'weight' in name:
                    torch.nn.init.uniform_(param, -uniform_mag, uniform_mag)

    #master.apply(init_weights)
    
    return master