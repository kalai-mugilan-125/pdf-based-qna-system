import torch
import torch.nn.functional as F

def hybrid_loss(qg_logits, decoder_labels, start_logits, end_logits, start_positions, end_positions, alpha=1.0, beta=1.0):
    qg_loss = F.cross_entropy(qg_logits.view(-1, qg_logits.size(-1)), decoder_labels.view(-1), ignore_index=-100)
    start_loss = F.cross_entropy(start_logits, start_positions)
    end_loss = F.cross_entropy(end_logits, end_positions)
    qa_loss = (start_loss + end_loss) / 2
    return alpha * qg_loss + beta * qa_loss