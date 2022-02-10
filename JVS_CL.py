import torch

""" --------- Loss function ----------- """
# normalized temperature-scaled cross entropy loss
# output 1 and output 2 is the 2 different versions of the same input image
def JVS_contrastive_loss(output1, output2, temperature):
    # concatenate v1 img and v2 img via the rows, stacking vertically
    # output1 and output2 should not be normalised
    # output1 = a batch of clip 1, output2 = a batch of clip 2
    # concatenate to do the calculation for both output1 and output2 at the same time
    """
    # assuming a batch of 2 for clip 1 and clip 2 respectively
    output1 = [x1,
               x2]

    output2 = [y1,
               y2]

    out = [x1,
           x2,
           y1,
           y2]
    """
    out = torch.cat([output1, output2], dim=0)

    # for the masking
    n_samples = len(out)
    #print('n_samples: ', n_samples)

    # Full similarity matrix
    # torch.mm --> matrix multiplication for tensors
    # when a transposed is done on a tensor, PyTorch doesn't generate new tensor with new layout,
    # it just modifies meta information in Tensor object so the offset and stride are for the new shape --> its memory
    # layout is different than a tensor of same shape made from scratch
    # contiguous --> makes a copy of tensor so the order of elements would be same as if tensor of same shape created from scratch
    # --> https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107
    # the diagonal of the matrix is the square of each vector element in the out vector, which shows the similarity between the same elements
    # need to get the diagonal for jvs
    """
    (4 x 1)
    out = [x1,
           x2,
           y1,
           y2]
           
    (1 x 4)
    out.t() = [x1, x2, y1, y2]
    
    (4 x 4)
    cov = [ (x1x1),  x1x2,   x1y1,    x1y2
            ------
             x2x1,  (x2x2),  x2y1,    x2y2
                    ------
             y1x1,   y1x2,  (y1y1),   y1y2
                            ------
             y2x1,   y2x2,    y2y1,  (y2y2)]
                                      -----
    
    The diag is the similarity of itself, and its sum is the denominator of the Jaccard Vector Similarity
    
    """
    cov = torch.mm(out, out.t().contiguous())
    #print('cov: ', cov)

    # take diagonal of cov --> make a vector
    # repeat the vector into the same size as cov (P)
    # H = P + P(transpose)
    # JVS_cov = cov/H
    # sim = torch.exp(JVS_cov / temperature)


    # formula for JVS = 2(x1 dot product x2)/(((x1 dot product x1)) + ((x2 dot product x2)))
    # diag = x1x1, x2x2, ... y1y1, y2y2, which is th denominator for the JVS
    """
    diag_cov = [x1x1, x2x2, y1y1, y2y2]
    """
    diag_cov = torch.diag(cov, 0)
    # reshape the diag so that the diag of the addition of it and its transpose will give a vector of x1.x1 + x2.x2 + ... y1.y1 + y2.y2
    diag_cov = torch.reshape(diag_cov, (diag_cov.size()[0], 1))
    #print('diag_cov: ', diag_cov)
    #print(diag_cov.size())

    """
    (4 x 1)
    diag_cov = [x1x1, 
                x2x2, 
                y1y1, 
                y2y2]
    (1 x 4)
    diag_cov_transpose = [[x1x1], [x2x2], [y1y1], [y2y2]]
    
    (4x4)
    H = diag_cov + diag_cov_transpose
    
      = [(x1x1 + x1x1),    x1x1 + x2x2,    x1x1 + y1y1,     x1x1 + y2y2
         -------------     
      
         x2x2 + x1x1,     (x2x2 + x2x2),    x2x2 + y1y1,     x2x2 + y2y2
                           ------------
         
         y1y1 + x1x1,     y1y1 + x2x2,    (y1y1 + y1y1),     y1y1 + y2y2
                                           ------------
         
         y2y2 + x1x1,     y2y2 + x2x2,    y2y2 + y1y1,     (y2y2 + y2y2)]
                                                            ------------
    
    The diag of H is the denominator for JVS
    """


    # transpose the diagonal
    diag_cov_transpose = torch.reshape(diag_cov, (1, diag_cov.size()[0]))
    #print('diag_cov_transpose: ',diag_cov_transpose)
    #print('transpose',diag_cov_transpose.size())

    H = diag_cov + diag_cov_transpose
    #print('H: ',H)
    #print(H.size())

    """
    # input dim should not be normalised if not the diagonal will just equals to 1 and it will be the cosine similarity 
    
    JVS_cov = cov / H
            
            = [ x1x1 / (x1x1 + x1x1),     x1x2 /  (x1x1 + x2x2),    x1y1 / (x1x1 + y1y1),    x1y2 / (x1x1 + y2y2)
                
                x2x1 / (x2x2 + x1x1),     x2x2 / (x2x2 + x2x2),     x2y1 / (x2x2 + y1y1),    x2y2 / (x2x2 + y2y2)
                    
                y1x1 / (y1y1 + x1x1),     y1x2 / (y1y1 + x2x2),     y1y1 / (y1y1 + y1y1),    y1y2 / (y1y1 + y2y2)
                            
                y2x1 / (y2y2 + x1x1),     y2x2 / (y2y2 + x2x2),     y2y1 / (y2y2 + y1y1),    y2y2 / (y2y2 + y2y2)]
                                
    """

    JVS_cov = cov / H

    # follows the numerator () formula for the contrastive loss
    sim = torch.exp(2*JVS_cov / temperature)
    print('sim: ', sim)
    #print(sim1.size())



    #sim = torch.exp(cov / temperature)
    #print('sim: ',sim)

    # Negative similarity
    # creates a 2-D tensor with True on the diagonal for the size of n_samples and False elsewhere
    """
    mask the diagonal because it shows the similarity between itself, e.g x1x1, x2x2, y1y1, y2y2
    
    mask = [(False),   True,       True,      True
            True,     (False),     True,      True
            True,      True,     (False),     True
            True,      True,       True,     (False)]
    
    
    """
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    #print('mask', mask)

    x = sim.masked_select(mask)
    print('x: ',x)

    x1 = sim.masked_select(mask).view(n_samples, -1)
    print('x1: ', x1)

    # Returns a new 1-D tensor which indexes the input tensor (sim) according to the boolean mask (mask) which is a BoolTensor.
    # returns a tensor with 1 row and n columns and sum it with the last dimension
    # masked_select(mask) --> only selects the 'True' value with respect to the mask and the similarity vector
    # .view(n_samples, -1) --> view the unmasked values into the rows with regards to the number of samples and whatever columns that fits
    # .sum(dim=1) --> sum the values in each row, which refers to the negative value for each sample, dim=0 is the outer array
    # neg = [x1 dot product other vectors, x2 dot product other vectors, y1 dot product other vectors, y2 dot product other vectors]
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=1)
    print('neg: ',neg)

    # Positive similarity
    # exp --> exponential of the sum of the last dimension after output1 * output2 divided by the temp
    """
    dim=0 is the outer array, dim=1 is the rows
    dot product is the sum of the multiplication of a(i)b(i) where i = 1
    (output1 * output2).sum(dim=1) = dot product
    Therefore, torch.exp(((2 * (output1 * output2).sum(dim=1)) / ((output1 * output1).sum(dim=1) + (output2 * output2).sum(dim=1)))/temperature)
    follows the JVS formula
    
    """

    pos = torch.exp(((2 * (output1 * output2).sum(dim=1)) / ((output1 * output1).sum(dim=1) + (output2 * output2).sum(dim=1)))/temperature)
    print('pos: ', pos)
    #print(pos.size())
    # concatenate via the rows, stacking vertically
    # there are 2 copies of the positive in 2 different denominators because the loss is symmetric
    # concatenate to get the symmetric loss function with respect to the negative value
    # e.g. neg[0] will be for x1, neg[2] will be for y1, neg[1] will be for x2, neg[3] will be for y2
    # the mean value will be the average loss for which the positions of the images are interchanged.
    # pos = [x1y1, x2y2, x1y1, x2y2]
    pos = torch.cat([pos, pos], dim=0)
    print(pos)

    # 2 copies of the numerator as the loss is symmetric but the denominator is 2 different values --> 1 for x, 1 for y
    # the loss will be a scalar value
    loss = -torch.log(pos / neg).mean()
    # print(loss)
    return loss



def JVS_loss(output1, output2):
    loss = (2 * (output1 * output2).sum(dim=1)) / ((output1 * output1).sum(dim=1) + (output2 * output2).sum(dim=1))
    #print(loss)
    batch_size = output1.size()[0]
    sum_batch_loss = sum(loss)
    #print(sum_batch_loss)
    avg_batch_loss = abs(sum_batch_loss/batch_size)
    #print(avg_batch_loss)
    return avg_batch_loss

x1 = torch.randn(5, 4)
#print(x1)
#print(x1.size())

x2 = torch.randn(5, 4)
#print(x2.size())
#print(x2)

x3 = torch.randn(128)


loss = JVS_contrastive_loss(output1=x1, output2=x2, temperature=0.5)
#print(loss)
JVS_loss(x1, x2)

#x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,34,36,37,38,39,40]
#y = x[0:32]
#print(len(y))
#print(y)



