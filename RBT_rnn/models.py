criterion = nn.NLLLoss()

#teacher
class Model1(nn.Module):
  def __init__(self, input_size, output_size, enc_hidden_size=256, dec_hidden_size=256):
    super(Model1, self).__init__()
    self.enc = EncoderRNN(input_size, enc_hidden_size)
    self.dec = DecoderRNN(dec_hidden_size, output_size)
    self.enc_optim = optim.SGD(self.enc.parameters(), lr=model1_lr)
    self.dec_optim = optim.SGD(self.dec.parameters(), lr=model1_lr)

  def enc_forward(self, input):
    encoder_hidden = self.enc.initHidden()
    input_length = input.size(0)
    encoder_outputs = torch.zeros(input_length, self.enc.hidden_size, device='cpu')
    
    for ei in range(input_length):
      encoder_output, encoder_hidden = self.enc(
          input[ei], encoder_hidden)
      encoder_outputs[ei] = encoder_output[0, 0]
    
    return encoder_hidden, encoder_outputs

  
  def dec_forward(self, target, encoder_hidden):
    target_length = target.size(0)
    decoder_input = torch.tensor([[SOS_token]], device='cpu') 
    decoder_hidden = encoder_hidden
    loss = 0
    for di in range(target_length):
        decoder_output, decoder_hidden = self.dec(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        loss += criterion(decoder_output, target[di])
        if decoder_input.item() == EOS_token:
            break
    return loss



class Model2(nn.Module):
  def __init__(self, input_size, output_size, enc_hidden_size=256, dec_hidden_size=256):
    super(Model2, self).__init__()
    self.enc = EncoderRNN(input_size, enc_hidden_size)
    self.dec = DecoderRNN(dec_hidden_size, output_size)

  def enc_forward(self, input):
    encoder_hidden = self.enc.initHidden()
    input_length = input.size(0)
    encoder_outputs = torch.zeros(input_length, self.enc.hidden_size, device='cpu')
    
    for ei in range(input_length):
      encoder_output, encoder_hidden = self.enc(
          input[ei], encoder_hidden)
      encoder_outputs[ei] = encoder_output[0, 0]
    
    return encoder_hidden, encoder_outputs

  
  def dec_forward(self, target, encoder_hidden):
    target_length = target.size(0)
    decoder_input = torch.tensor([[SOS_token]], device='cpu') 
    decoder_hidden = encoder_hidden
    loss = 0
    for di in range(target_length):
        decoder_output, decoder_hidden = self.dec(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  
        loss += criterion(decoder_output, target[di])
        if decoder_input.item() == EOS_token:
            break
    return loss


  