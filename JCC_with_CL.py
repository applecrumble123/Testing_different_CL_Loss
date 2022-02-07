import torch



""" --------- Loss function ----------- """
def JCC_contrastive_loss(output1, output2, temperature):
    # output1 = a batch of clip 1, output2 = a batch of clip 2
    # concatenate to do the calculation for both output1 and output2 at the same time
    """
    # assuming a batch of 2 for clip 1 and clip 2 respectively
    output1 = [[x1a, x1b],  ---> output1 = [[x1],
               [x2a, x2b]]                 [x2]]

    output2 = [[y1a, y1b],  ---> output2 = [[y1],
               [y2a, y2b]]                  [y2]]

    """

    output1 = output1.view(-1, output1.shape[-1])
    output2 = output2.view(-1, output2.shape[-1])


    print('output1: ', output1)
    print('output2: ', output2)

    """
    concat_output1_output2 = [[x1a, x1b], 
                              [x2a, x2b],
                              [y1a, y1b],
                              [y2a, y2b]]             
                              
                           = [[x1],
                              [x2],
                              [y1],
                              [y2]]
    
    concat_output1_output2_transpose = [[x1a, x2a, y1a, y2a],
                                        [x1b, x2b, y2b, y2b]]             
                              
                           
    """

    concat_output1_output2 = torch.cat([output1, output2], dim=0)
    print('concat_output1_output2: ', concat_output1_output2)

    # for the masking
    n_samples = len(concat_output1_output2)
    #print('n_samples: ', n_samples)

    concat_output1_output2_transpose = concat_output1_output2.t()
    print('concat_output1_output2_transpose: ', concat_output1_output2_transpose)

    """
    (4 x 1)
    out = [x1,
           x2,
           y1,
           y2]
           
    (1 x 4)
    out.t() = [x1, x2, y1, y2]
    
    (4 x 4)
    
    cov = [ E[x1x1],  E[x1x2],   E[x1y1],    E[x1y2]
            -------
            E[x2x1],  E[x2x2],   E[x2y1],    E[x2y2]
                      -------
            E[y1x1],  E[y1x2],   E[y1y1],    E[y1y2]
                                 -------
            E[y2x1],  E[y2x2],   E[y2y1],    E[y2y2]]
                                              ------
    
    The diag is the similarity of itself, the rest of the elements are the numerators for JCC loss.
    
    The dot product is the sum of x1 * x1, which is the E[x1x1]. 
    
    There is no need to divide by total n-elements after doing the sum for each row as it cancels out in the denominator in the equation
    """

    # Full similarity matrix
    # torch.mm --> matrix multiplication for tensors
    # when a transposed is done on a tensor, PyTorch doesn't generate new tensor with new layout,
    # it just modifies meta information in Tensor object so the offset and stride are for the new shape --> its memory
    # layout is different than a tensor of same shape made from scratch
    # contiguous --> makes a copy of tensor so the order of elements would be same as if tensor of same shape created from scratch
    # --> https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107

    cov = torch.mm(concat_output1_output2, concat_output1_output2_transpose.contiguous())
    print('cov: ', cov)

    """
    output1_transpose = [[x1a, x2a],   ---> output1_transpose = [[ x1'],
                         [x1b, x2b]]                             [ x2']]

    output1_transpose = [[y1a, y2a],   ---> output2_transpose = [[ y1'],
                         [y1b, y2b]]                             [ y2']]
    """
    output1_transpose = output1.transpose(dim0=1, dim1=0)
    output2_transpose = output2.transpose(dim0=1, dim1=0)

    print('output1_transpose: ', output1_transpose)
    print('output2_transpose: ', output2_transpose)

    """
    output1 x output1_transpose = [[x1a, x1b]    x    [[x1a, x2a],   
                                   [x2a, x2b]]         [x1b, x2b]]
                                    
                                = [[x1]]         x     [[x1']]
                                
                                = [[x1x1']]  
    
    E[output1 x output1_transpose] = [[x1x1']].sum(dim=1)
    
    
    output2 x output2_transpose = [[y1a, y1b]    x    [[y1a, y2a],   
                                   [y2a, y2b]]         [y1b, y2b]]
                                    
                                = [[y1]]         x     [[y1']]
                                
                                = [[y1y1']]  
    
    E[output2 x output2_transpose] = [[y1y1']].sum(dim=1)
    """

    # row wise mean as it is per batch
    # -1 gives the last dim, which is the row
    # sum per row due to per batch, each row represents the batch
    # there is no need to divide by total n-elements after doing the sum for each row as it cancels out in the denominator in the equation
    # in the denominator --> E[a] + E[b] = E[a+b] --> also divided by n in the denominator
    # numerator is divided by n and the denminator is divided by n --> cancels out since both is divided by n
    # e.g. 4/2 = 2/1
    expectation_output1_output1_transpose = output1.mm(output1_transpose).sum(dim=1)
    expectation_output2_output2_transpose = output2.mm(output2_transpose).sum(dim=1)

    print('expectation_output1_output1_transpose: ', expectation_output1_output1_transpose)
    print('expectation_output2_output2_transpose: ', expectation_output2_output2_transpose)

    """
    out = [E[x1x1'], 
           E[x2x2'], 
           E[y1y1'], 
           E[y2y2']]
        
        = [a1,
           a2,
           b1,
           b2] 
           
    shape = (4,0)
    """

    out = torch.cat([expectation_output1_output1_transpose, expectation_output2_output2_transpose], dim=0)
    print('out: ', out)

    """
    out_transpose = [[E[x1x1']], 
                     [E[x2x2']], 
                     [E[y1y1']], 
                     [E[y2y2']]]
                  
                  = [[a1],
                     [a2],
                     [b1],
                     [b2]] 
                     
    shape = (4,1)
    """
    out_transpose = torch.reshape(out, (-1, 1))
    print('out_transpose: ', out_transpose)

    """
    H = [E[x1x1'],       [[E[x1x1']],
         E[x2x2'],   +    [E[x2x2']],
         E[y1y1'],        [E[y1y1']],
         E[y2y2']]        [E[y2y2']]]
         
      
      = [E[x1x1'] + E[x1x1'],   E[x1x1'] + E[x2x2'],   E[x1x1'] + E[y1y1'],   E[x1x1'] + E[y2y2']
         -------------------
         
         E[x2x2'] + E[x1x1'],   E[x2x2'] + E[x2x2'],   E[x2x2'] + E[y1y1'],   E[x2x2'] + E[y2y2']
                                -------------------
                                
         E[y1y1'] + E[x1x1'],   E[y1y1'] + E[x2x2'],   E[y1y1'] + E[y1y1'],   E[y1y1'] + E[y2y2']
                                                       -------------------
                                                       
         E[y2y2'] + E[x1x1'],   E[y2y2'] + E[x2x2'],   E[y2y2'] + E[y1y1'],   E[y2y2'] + E[y2y2']]
                                                                              -------------------  
    """

    H = out + out_transpose

    print('H: ', H)

    """
    JCC_cov = cov / H
    
            = [E[x1x1] / E[x1x1'] + E[x1x1'],   E[x1x2]/ E[x1x1'] + E[x2x2'],    E[x1y1] / E[x1x1'] + E[y1y1'],   E[x1y2] / E[x1x1'] + E[y2y2']
               -----------------------------
             
               E[x2x1] / E[x2x2'] + E[x1x1'],   E[x2x2] / E[x2x2'] + E[x2x2'],   E[x2y1] / E[x2x2'] + E[y1y1'],   E[x2y2] / E[x2x2'] + E[y2y2']
                                                -----------------------------
                                    
               E[y1x1] / E[y1y1'] + E[x1x1'],   E[y1x2] / E[y1y1'] + E[x2x2'],   E[y1y1] / E[y1y1'] + E[y1y1'],   E[y1y2] / E[y1y1'] + E[y2y2']
                                                                                  -----------------------------
                                                           
               E[y2x1]/ E[y2y2'] + E[x1x1'],    E[y2x2] / E[y2y2'] + E[x2x2'],   E[y2y1] / E[y2y2'] + E[y1y1'],   E[y2y2] / E[y2y2'] + E[y2y2']]
                                                                                                                  ------------------------------  
    
    """

    JCC_cov = cov / H

    print('JCC_cov: ', JCC_cov)

    # follows the numerator () formula for the contrastive loss
    sim = torch.exp(-2 * JCC_cov / temperature)
    print('sim: ',sim)

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

    print('mask: ',mask)

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

    print('neg: ', neg)



    # Positive similarity
    # exp --> exponential of the sum of the last dimension after output1 * output2 divided by the temp
    """
    dim=0 is the outer array, dim=1 is the rows
    dot product is the sum of the multiplication of a(i)b(i) where i = 1
    (output1 * output2).sum(dim=1) = dot product
    Therefore, torch.exp(((2 * (output1 * output2).sum(dim=1)) / ((output1 * output1).sum(dim=1) + (output2 * output2).sum(dim=1)))/temperature)
    follows the JVS formula

    """
    output1_output_2 = output1.transpose(dim0=1, dim1=0).mm(output2).sum(dim=1)
    print('output1_output_2: ',output1.transpose(dim0=1, dim1=0).mm(output2))
    print(output1)

    x = output1.transpose(dim0=1, dim1=0)
    print('x: ', x)
    print('x_transpose: ', x.transpose(dim0=1, dim1=0))

    y = output2.transpose(dim0=1, dim1=0)
    print('y: ', y)


    z = x.transpose(dim0=1, dim1=0).mm(y)
    print('z: ', z)
    print(z.size())


    output1_output1 = output1.transpose(dim0=1, dim1=0).mm(output1).sum(dim=1)

    output2_output2 = output2.transpose(dim0=1, dim1=0).mm(output2).sum(dim=1)
    sumv = output1_output1 + output2_output2
    mix = output1_output_2 / sumv
    # the transpose will give a cov matrix (n x n)
    # n square is the mean for the cov matrix (n x n) --> that is why it is 1/n^2, it is an element wise mean
    # therefore, it is the mean of the matrix
    pos = torch.exp(-2 * mix)




    #pos = torch.exp(((2 * (output1 * output2).sum(dim=1)) / ((output1 * output1).sum(dim=1) + (output2 * output2).sum(dim=1))) / temperature)
    print('pos: ', pos)
    # print(pos.size())
    # concatenate via the rows, stacking vertically
    # there are 2 copies of the positive in 2 different denominators because the loss is symmetric
    # concatenate to get the symmetric loss function with respect to the negative value
    # e.g. neg[0] will be for x1, neg[2] will be for y1, neg[1] will be for x2, neg[3] will be for y2
    # the mean value will be the average loss for which the positions of the images are interchanged.
    # pos = [x1y1, x2y2, x1y1, x2y2]
    #pos = torch.cat([pos, pos], dim=0)
    #print(pos)

    # 2 copies of the numerator as the loss is symmetric but the denominator is 2 different values --> 1 for x, 1 for y
    # the loss will be a scalar value
    loss = -torch.log(pos / neg).mean()
    print(loss)

    #return loss


# batch 4 with 128 features
#x1 = torch.randn(5, 3)
#print(y)
x1 = torch.tensor([[1,2,3,4], [3,4,5,6]])
#x1 = torch.tensor([[1,2]])
#print(x1)
#print(x1.size())


# batch 4 with 128 features
#x2 = torch.randn(5, 3)
x2 = torch.tensor([[5,6,7,8], [7,8,9,10]])
#x2 = torch.tensor([[5,6]])

JCC_contrastive_loss(x1, x2, temperature=0.5)