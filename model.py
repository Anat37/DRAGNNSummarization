from decoder import *
from data import *
from pointerTBRU import BeamSearchProviderTBRU, LSTMTBRU, AttentionTBRU, ConcatTBRU, PointerTBRU, ContextTBRU, CoverageAttentionTBRU, ConcatStateTBRU

def build_decoder_model(mask_layer=None):
    hidden_dim = 256
    emb_dim= 128
    
    master = DRAGNNDecoderMaster()
    
    embeddings_computer = EmbeddingComputer(VOCAB_SIZE + ADDITIONAL_WORDS, emb_dim, PAD_TOKEN_ID)
    master.add_component_encoder(TBRU("embed", TaggerRecurrent("input", "embed", False), embeddings_computer, (1,), True).cuda())
    #master.add_component(TBRU("extractive", TaggerRecurrent("embed", "extractive"), TaggerComputer(1000, 1000), (1,), True).cuda())
    master.add_component_encoder(TBRU("bilstm", RNNSolidRecurrent("embed", "bilstm"), BILSTMSolidComputer(emb_dim, hidden_dim), (1,), True).cuda())
    master.add_component_encoder(TBRU("bilstm_reduced", TaggerRecurrent("bilstm", "bilstm_reduced", False), TaggerComputer(hidden_dim * 2, hidden_dim), (1,), is_solid=True).cuda())
    
    master.add_component_decoder(TBRU("decoder_embed", TaggerRecurrent(None, "decoder_embed", True),embeddings_computer, (1,), is_solid=True).cuda())
    
    master.add_component_decoder(BeamSearchProviderTBRU("beamProvider", None, 0, "decoder", True, is_solid=False))
    master.add_component_decoder(BeamSearchProviderTBRU("beamProvider1", None, 0, "lstm_state_layer", True, is_solid=False))
    master.add_component_decoder(BeamSearchProviderTBRU("beamProvider2", None, 1, "coverage_layer", True, is_solid=False))
    master.add_component_decoder(BeamSearchProviderTBRU("beamProvider3", None, 0, "context", True, is_solid=False))
    
    master.add_component_decoder(ConcatStateTBRU("decoder_input_concat", False, "decoder_embed", 'context', hidden_dim, False, solid_modifiable=False))
    
    master.add_component_decoder(TBRU("decoder_input", TaggerRecurrent("decoder_input_concat", "decoder_input", False), TaggerComputer(emb_dim + hidden_dim, hidden_dim), (1,), is_solid=False, solid_modifiable=False).cuda())
    pTBRU = LSTMTBRU("decoder", "lstm_state_layer", False, hidden_dim, hidden_dim, "bilstm_reduced", "decoder_input", False, solid_modifiable=False)
    master.add_component_decoder(pTBRU.cuda())
    
    master.add_component_decoder(ConcatStateTBRU('attention_input', False, 'decoder', 'lstm_state_layer', hidden_dim, is_first=False, solid_modifiable=False))
    
    master.add_component_decoder(CoverageAttentionTBRU("attention_layer", "coverage_layer", False, hidden_dim, 2*hidden_dim, hidden_dim, 'attention_input','bilstm_reduced', is_first=False, mask_layer=None, solid_modifiable=False).cuda())
    master.add_component_decoder(ContextTBRU('context', False, 'attention_layer', 'bilstm_reduced', is_first=False, solid_modifiable=False))
    master.add_component_decoder(ConcatTBRU('context_concat', False, 'decoder', 'context', is_first=False, solid_modifiable=False))
    master.add_component_decoder(TBRU("output", TaggerRecurrent("context_concat", "output", False), TaggerComputer(2 * hidden_dim, VOCAB_SIZE), (1,), is_solid=False, solid_modifiable=False).cuda())
    master.add_component_decoder(PointerTBRU("pointer_final", False, 4*hidden_dim, VOCAB_SIZE + ADDITIONAL_WORDS, "attention_layer", "context", "output", "decoder_input", "input", "decoder","lstm_state_layer", solid_modifiable=False).cuda())
    return master