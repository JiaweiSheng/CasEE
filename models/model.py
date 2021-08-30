from models.layers import *


class TypeCls(nn.Module):
    def __init__(self, config):
        super(TypeCls, self).__init__()
        self.type_emb = nn.Embedding(config.type_num, config.hidden_size)
        self.register_buffer('type_indices', torch.arange(0, config.type_num, 1).long())
        self.dropout = nn.Dropout(config.decoder_dropout)

        self.config = config
        self.Predictor = AdaptiveAdditionPredictor(config.hidden_size, dropout_rate=config.decoder_dropout)

    def forward(self, text_rep, mask):
        type_emb = self.type_emb(self.type_indices)
        pred = self.Predictor(type_emb, text_rep, mask)  # [b, c]
        p_type = torch.sigmoid(pred)
        return p_type, type_emb


class TriggerRec(nn.Module):
    def __init__(self, config, hidden_size):
        super(TriggerRec, self).__init__()
        self.ConditionIntegrator = ConditionalLayerNorm(hidden_size)
        self.SA = MultiHeadedAttention(hidden_size, heads_num=config.decoder_num_head, dropout=config.decoder_dropout)

        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.head_cls = nn.Linear(hidden_size, 1, bias=True)
        self.tail_cls = nn.Linear(hidden_size, 1, bias=True)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.config = config

    def forward(self, query_emb, text_emb, mask):
        '''

        :param query_emb: [b, e]
        :param text_emb: [b, t, e]
        :param mask: 0 if masked
        :return: [b, t, 1], [], []
        '''

        h_cln = self.ConditionIntegrator(text_emb, query_emb)

        h_cln = self.dropout(h_cln)
        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        h_sa = self.dropout(h_sa)
        inp = self.layer_norm(h_sa + h_cln)
        inp = gelu(self.hidden(inp))
        inp = self.dropout(inp)
        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, 1]
        p_e = torch.sigmoid(self.tail_cls(inp))  # [b, t, 1]
        return p_s, p_e, h_cln


class ArgsRec(nn.Module):
    def __init__(self, config, hidden_size, num_labels, seq_len, pos_emb_size):
        super(ArgsRec, self).__init__()
        self.relative_pos_embed = nn.Embedding(seq_len * 2, pos_emb_size)
        self.ConditionIntegrator = ConditionalLayerNorm(hidden_size)
        self.SA = MultiHeadedAttention(hidden_size, heads_num=config.decoder_num_head, dropout=config.decoder_dropout)
        self.hidden = nn.Linear(hidden_size + pos_emb_size, hidden_size)

        self.head_cls = nn.Linear(hidden_size, num_labels, bias=True)
        self.tail_cls = nn.Linear(hidden_size, num_labels, bias=True)

        self.gate_hidden = nn.Linear(hidden_size, hidden_size)
        self.gate_linear = nn.Linear(hidden_size, num_labels)

        self.seq_len = seq_len
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.config = config

    def forward(self, text_emb, relative_pos, trigger_mask, mask, type_emb):
        '''
        :param query_emb: [b, 4, e]
        :param text_emb: [b, t, e]
        :param relative_pos: [b, t, e]
        :param trigger_mask: [b, t]
        :param mask:
        :param type_emb: [b, e]
        :return:  [b, t, a], []
        '''
        trigger_emb = torch.bmm(trigger_mask.unsqueeze(1).float(), text_emb).squeeze(1)  # [b, e]
        trigger_emb = trigger_emb / 2

        h_cln = self.ConditionIntegrator(text_emb, trigger_emb)
        h_cln = self.dropout(h_cln)
        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        h_sa = self.dropout(h_sa)
        h_sa = self.layer_norm(h_sa + h_cln)

        rp_emb = self.relative_pos_embed(relative_pos)
        rp_emb = self.dropout(rp_emb)

        inp = torch.cat([h_sa, rp_emb], dim=-1)

        inp = gelu(self.hidden(inp))
        inp = self.dropout(inp)

        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, l]
        p_e = torch.sigmoid(self.tail_cls(inp))

        type_soft_constrain = torch.sigmoid(self.gate_linear(type_emb))  # [b, l]
        type_soft_constrain = type_soft_constrain.unsqueeze(1).expand_as(p_s)
        p_s = p_s * type_soft_constrain
        p_e = p_e * type_soft_constrain

        return p_s, p_e, type_soft_constrain


class CasEE(nn.Module):
    def __init__(self, config, model_weight, pos_emb_size):
        super(CasEE, self).__init__()
        self.bert = model_weight

        self.config = config
        self.args_num = config.args_num
        self.text_seq_len = config.seq_length

        self.type_cls = TypeCls(config)
        self.trigger_rec = TriggerRec(config, config.hidden_size)
        self.args_rec = ArgsRec(config, config.hidden_size, self.args_num, self.text_seq_len, pos_emb_size)
        self.dropout = nn.Dropout(config.decoder_dropout)

        self.loss_0 = nn.BCELoss(reduction='none')
        self.loss_1 = nn.BCELoss(reduction='none')
        self.loss_2 = nn.BCELoss(reduction='none')

    def forward(self, tokens, segment, mask, type_id, type_vec, trigger_s_vec, trigger_e_vec, relative_pos, trigger_mask, args_s_vec, args_e_vec, args_mask):
        '''

        :param tokens: [b, t]
        :param segment: [b, t]
        :param mask: [b, t], 0 if masked
        :param trigger_s: [b, t]
        :param trigger_e: [b, t]
        :param relative_pos:
        :param trigger_mask: [0000011000000]
        :param args_s: [b, l, t]
        :param args_e: [b, l, t]
        :param args_m: [b, k]
        :return:
        '''

        outputs = self.bert(
            tokens,
            attention_mask=mask,
            token_type_ids=segment,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
        )

        output_emb = outputs[0]
        p_type, type_emb = self.type_cls(output_emb, mask)
        p_type = p_type.pow(self.config.pow_0)
        type_loss = self.loss_0(p_type, type_vec)
        type_loss = torch.sum(type_loss)

        type_rep = type_emb[type_id, :]
        p_s, p_e, text_rep_type = self.trigger_rec(type_rep, output_emb, mask)
        p_s = p_s.pow(self.config.pow_1)
        p_e = p_e.pow(self.config.pow_1)
        p_s = p_s.squeeze(-1)
        p_e = p_e.squeeze(-1)
        trigger_loss_s = self.loss_1(p_s, trigger_s_vec)
        trigger_loss_e = self.loss_1(p_e, trigger_e_vec)
        mask_t = mask.float()  # [b, t]
        trigger_loss_s = torch.sum(trigger_loss_s.mul(mask_t))
        trigger_loss_e = torch.sum(trigger_loss_e.mul(mask_t))

        p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, relative_pos, trigger_mask, mask, type_rep)
        p_s = p_s.pow(self.config.pow_2)
        p_e = p_e.pow(self.config.pow_2)
        args_loss_s = self.loss_2(p_s, args_s_vec.transpose(1, 2))  # [b, t, l]
        args_loss_e = self.loss_2(p_e, args_e_vec.transpose(1, 2))
        mask_a = mask.unsqueeze(-1).expand_as(args_loss_s).float()  # [b, t, l]
        args_loss_s = torch.sum(args_loss_s.mul(mask_a))
        args_loss_e = torch.sum(args_loss_e.mul(mask_a))

        trigger_loss = trigger_loss_s + trigger_loss_e
        args_loss = args_loss_s + args_loss_e

        type_loss = self.config.w1 * type_loss
        trigger_loss = self.config.w2 * trigger_loss
        args_loss = self.config.w3 * args_loss
        loss = type_loss + trigger_loss + args_loss
        return loss, type_loss, trigger_loss, args_loss

    def plm(self, tokens, segment, mask):
        assert tokens.size(0) == 1

        outputs = self.bert(
            tokens,
            attention_mask=mask,
            token_type_ids=segment,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
        )
        output_emb = outputs[0]
        return output_emb

    def predict_type(self, text_emb, mask):
        assert text_emb.size(0) == 1
        p_type, type_emb = self.type_cls(text_emb, mask)
        p_type = p_type.view(self.config.type_num).data.cpu().numpy()
        return p_type, type_emb

    def predict_trigger(self, type_rep, text_emb, mask):
        assert text_emb.size(0) == 1
        p_s, p_e, text_rep_type = self.trigger_rec(type_rep, text_emb, mask)
        p_s = p_s.squeeze(-1)  # [b, t]
        p_e = p_e.squeeze(-1)
        mask = mask.float()  # [1, t]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.text_seq_len).data.cpu().numpy()  # [b, t]
        p_e = p_e.view(self.text_seq_len).data.cpu().numpy()
        return p_s, p_e, text_rep_type

    def predict_args(self, text_rep_type, relative_pos, trigger_mask, mask, type_rep):
        assert text_rep_type.size(0) == 1
        p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, relative_pos, trigger_mask, mask, type_rep)
        mask = mask.unsqueeze(-1).expand_as(p_s).float()  # [b, t, l]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.text_seq_len, self.args_num).data.cpu().numpy()
        p_e = p_e.view(self.text_seq_len, self.args_num).data.cpu().numpy()
        return p_s, p_e, type_soft_constrain
