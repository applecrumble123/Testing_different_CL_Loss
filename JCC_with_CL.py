import torch
import math



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

    shape --> (2,2)
    """

    output1 = output1.view(-1, output1.shape[-1])
    output2 = output2.view(-1, output2.shape[-1])

    #print('output1: ', output1)
    #print('output2: ', output2)

    """
    concat_output1_output2 = [[x1a, x1b],
                              [x2a, x2b],
                              [y1a, y1b],
                              [y2a, y2b]]   
                            
                           = [x1, 
                              x2, 
                              y1, 
                              y2]
    
    shape --> e.g. (4,2)          
    """

    concat_output1_output2 = torch.cat([output1, output2], dim=0)
    #print('concat_output1_output2: ', concat_output1_output2)
    #print(concat_output1_output2.size())

    # for the masking
    n_samples = len(concat_output1_output2)
    # print('n_samples: ', n_samples)

    """
    concat_reshape = [[[x1a, x1b]],
                      [[x2a, x2b]],
                      [[y1a, y1b]],
                      [[y2a, y2b]]] 
    
    2 * 2 batch of 1 output with 2 features        
    shape --> e.g. (4 x 1 x 2)
    """

    concat_reshape = torch.reshape(concat_output1_output2, (concat_output1_output2.shape[0], 1, concat_output1_output2.shape[1]))
    #print('concat_reshape: ',concat_reshape)
    #print(concat_reshape.size())


    """
    transpose the output with features while keeping the number of batches the same
    concat_reshape_transpose = [[[x1a], [x1b]],
                                [[x2a], [x2b]],
                                [[y1a], [y1b]],
                                [[y2a], [y2b]]] 
    
    shape --> e.g. (4 x 2 x 1)
    """

    # change the dim1 and dim2 position
    concat_reshape_transpose = concat_reshape.permute(0,2,1)
    #print('concat_reshape_transpose: ', concat_reshape_transpose)
    #print(concat_reshape_transpose.size())

    """
    get the d*d cross correlation feature dimension matrix
    
    concat_reshape_transpose *  concat_reshape = [[[x1a], [x1b]],       [[[x1a, x1b]],
                                                  [[x2a], [x2b]],   *    [[x2a, x2b]],
                                                  [[y1a], [y1b]],        [[y1a, y1b]],
                                                  [[y2a], [y2b]]]        [[y2a, y2b]]]
    
    
    cross_corr = [[[x1ax1a, x1ax1b],
                  [x1bx1a, x1bx1b]],
         
                  [[x2ax2a, x2ax2b], 
                   [x2bx2a, x2bx2b]],
          
                  [[y1ay1a, y1ay1b], 
                   [y1ay1b, y1by1b]],
          
                   [[y2ay2a, y2ay2b],
                    [y2by2a, y2by2b]]]
       
              = [[E[x1'x1]],
         
                 [E[x2'x2]],
          
                 [E[y1'y1]],
          
                 [E[y2'y2]]]
    
    shape --> e.g. (4 x 2 x 2)
       
    """
    cross_corr = concat_reshape_transpose.bmm(concat_reshape)
    #print('cross_corr: ', cross_corr)
    #print(cross_corr.size())

    """
    reshape the matrix to do the sum for the denominator of the JCC loss
    
    cross_corr_reshape = [[[x1ax1a, x1ax1b, x1bx1a, x1bx1b]],
         
                          [[x2ax2a, x2ax2b, x2bx2a, x2bx2b]],
          
                          [[y1ay1a, y1ay1b, y1ay1b, y1by1b]],
          
                          [[y2ay2a, y2ay2b, y2by2a, y2by2b]]]

    
    shape --> e.g. (4 x 1 x 4)
    """
    cross_corr_reshape = torch.reshape(cross_corr, (cross_corr.shape[0], 1, cross_corr.shape[1] * cross_corr.shape[2]))
    #print('cross_corr_reshape: ', cross_corr_reshape)
    #print(cross_corr_reshape.size())

    """
    cross_corr_reshape_2 = [[x1ax1a, x1ax1b, x1bx1a, x1bx1b],
         
                            [x2ax2a, x2ax2b, x2bx2a, x2bx2b],
          
                            [y1ay1a, y1ay1b, y1ay1b, y1by1b],
          
                            [y2ay2a, y2ay2b, y2by2a, y2by2b]]
                         
                         =  [E[x1'x1], E[x2'x2], E[y1'y1], E[y2'y2]]
    
    shape --> e.g. (4 x 4)
    """

    cross_corr_reshape_2 = torch.reshape(cross_corr, (cross_corr.shape[0], cross_corr.shape[1] * cross_corr.shape[2]))
    #print('cross_corr_reshape_2: ', cross_corr_reshape_2)
    #print(cross_corr_reshape_2.size())

    """
    cross_corr_reshape_concat = [[[x1ax1a, x1ax1b, x1bx1a, x1bx1b],
                                  [x1ax1a, x1ax1b, x1bx1a, x1bx1b],
                                  [x1ax1a, x1ax1b, x1bx1a, x1bx1b],
                                  [x1ax1a, x1ax1b, x1bx1a, x1bx1b]],
         
                                 [[x2ax2a, x2ax2b, x2bx2a, x2bx2b],
                                  [x2ax2a, x2ax2b, x2bx2a, x2bx2b],
                                  [x2ax2a, x2ax2b, x2bx2a, x2bx2b],
                                  [x2ax2a, x2ax2b, x2bx2a, x2bx2b]],
          
                                 [[y1ay1a, y1ay1b, y1ay1b, y1by1b],
                                  [y1ay1a, y1ay1b, y1ay1b, y1by1b],
                                  [y1ay1a, y1ay1b, y1ay1b, y1by1b],
                                  [y1ay1a, y1ay1b, y1ay1b, y1by1b]],
          
                                 [[y2ay2a, y2ay2b, y2by2a, y2by2b],
                                  [y2ay2a, y2ay2b, y2by2a, y2by2b],
                                  [y2ay2a, y2ay2b, y2by2a, y2by2b],
                                  [y2ay2a, y2ay2b, y2by2a, y2by2b]]]
                        
                            = [[E[x1'x1], E[x1'x1], E[x1'x1], E[x1'x1]],
         
                               [E[x2'x2], E[x2'x2], E[x2'x2], E[x2'x2]],
          
                               [E[y1'y1], E[y1'y1], E[y1'y1], E[y1'y1]],
          
                               [E[y2'y2], E[y2'y2], E[y2'y2], E[y2'y2]]]
    
    shape --> e.g. (4 x 4 x 4)
    """

    cross_corr_reshape_concat = torch.cat([cross_corr_reshape] * n_samples, dim=1)
    #print('cross_corr_concat: ', cross_corr_reshape_concat)
    #print(cross_corr_reshape_concat.size())

    """
    broadcasting summation of matrix
    
    sum of the col in each row to get the sum of expectations for the denominator
    
    sum_cross_corr = [[E[x1'x1], E[x1'x1], E[x1'x1], E[x1'x1]],        [E[x1'x1], E[x2'x2], E[y1'y1], E[y2'y2]]
         
                      [E[x2'x2], E[x2'x2], E[x2'x2], E[x2'x2]],    +    
                        
                      [E[y1'y1], E[y1'y1], E[y1'y1], E[y1'y1]],        
                       
                      [E[y2'y2], E[y2'y2], E[y2'y2], E[y2'y2]]]
                      
                    
                    = [[E[x1'x1] + E[x1'x1],    E[x1'x1] + E[x2'x2],    E[x1'x1] + E[y1'y1],    E[x1'x1] +  E[y2'y2]], 
                        -------------------  
         
                       [E[x2'x2] + E[x1'x1],    E[x2'x2] + E[x2'x2],    E[x2'x2] + E[y1'y1],    E[x2'x2] +  E[y2'y2]],   
                                                -------------------  
                        
                       [E[y1'y1] + E[x1'x1],    E[y1'y1] + E[x2'x2],    E[y1'y1] + E[y1'y1],    E[y1'y1] +  E[y2'y2]],  
                                                                        -------------------       
                       
                       [E[y2'y2] + E[x1'x1],    E[y2'y2] + E[x2'x2],    E[y2'y2] + E[y1'y1],    E[y2'y2] +  E[y2'y2]]] 
                                                                                                 -------------------
    
    shape --> e.g. (4 x 4)
    
    The diag is the similarity of itself, the rest of the elements are the denominator for JCC loss.   
    
    There is no need to divide by total n-elements after doing the sum for each row as it cancels out in the numerator in the equation      
    """
    # Full similarity matrix
    sum_cross_corr = (cross_corr_reshape_concat + cross_corr_reshape_2).sum(dim=-1)

    #print('sum_cross_corr: ', sum_cross_corr)
    #print(sum_cross_corr.size())

    """
    transpose the concat reshape
    
    concat_reshape --> transpose_concat_reshape
    
    [[[x1]],       [[[x1']],

     [[x2]],  -->   [[x2']],
    
     [[y1]],        [[y1']],
             
     [[y2]]]        [[y2']]]
     
     shape --> e.g. (4 x 1 x 2) to (4 x 2 x 1)
    """

    transpose_concat_reshape = concat_reshape.permute(0,2,1)
    #print('transpose_concat_reshape: ', transpose_concat_reshape)
    #print(transpose_concat_reshape.size())

    """
    reshape transpose_concat_reshape to so that it can be concatenated at the features dimension
    
    transpose_concat_reshape_2 = [ [ [x1'], [x2'], [y1'], [y2'] ] ]
    
    shape --> e.g. (4 x 1 x 1 x 2)
    """

    transpose_concat_reshape_2 = torch.reshape(transpose_concat_reshape, (transpose_concat_reshape.shape[0], 1, transpose_concat_reshape.shape[1], transpose_concat_reshape.shape[2]))
    #print('transpose_concat_reshape_2: ', transpose_concat_reshape_2)
    #print(transpose_concat_reshape_2.size())

    """
    concat the transpose_concat_reshape_2 at each row where it contains [x1'], [x2'], [y1'], [y2'] to the number of samples
    
    concat_transpose_concat_reshape_2 = [ [ [x1'], [x2'], [y1'], [y2'] ], 
                                          [ [x1'], [x2'], [y1'], [y2'] ],
                                          [ [x1'], [x2'], [y1'], [y2'] ],
                                          [ [x1'], [x2'], [y1'], [y2'] ] ]
                                          
    shape --> e.g. (4 x 4 x 1 x 2)                                      
    """

    concat_transpose_concat_reshape_2 = torch.cat([transpose_concat_reshape_2] * n_samples, dim=1)
    #print('concat_transpose_concat_reshape_2: ', concat_transpose_concat_reshape_2)
    #print(concat_transpose_concat_reshape_2.size())


    """
    broadcasting multiplication of matrix
    
    Multiply the transpose matrix with the original matrix
    
    numerator = concat_transpose_concat_reshape_2 * concat_reshape
              
              = [ [ [x1'], [x2'], [y1'], [y2'] ],       [ [x1], [x2], [y1], [y2] ]
                  [ [x1'], [x2'], [y1'], [y2'] ],   * 
                  [ [x1'], [x2'], [y1'], [y2'] ],
                  [ [x1'], [x2'], [y1'], [y2'] ] ]
                  
              = [ [ [x1'x1], [x2'x1], [y1'x1], [y2'x1] ],   
                    -------
                 
                  [ [x1'x2], [x2'x2], [y1'x2], [y2'x2] ],  
                             -------
                  
                  [ [x1'y1], [x2'y1], [y1'y1], [y2'y1] ],
                                      -------
                  
                  [ [x1'y2], [x2'y2], [y1'y2], [y2'y2] ] ]
                                               -------
                                               
    shape --> e.g. (4 x 4 x 2 x 1)  *  (4 x 1 x 2)  = (4 x 4 x 2 x 2)          
    
    The diag is the similarity of itself, the rest of the elements are the numerator for JCC loss.   
    
    There is no need to divide by total n-elements after doing the sum for each row as it cancels out in the denominator in the equation  
    
    The sum of each d*d feature dimension cross correlation matrix is the expecation                  
    """

    numerator = concat_transpose_concat_reshape_2 * concat_reshape
    #print('numerator: ', numerator)
    #print(numerator.size())

    """
    sum the d*d feature dimension cross correlation matrix
    """
    cross_corr_numerator = numerator.sum(dim=-1).sum(dim=-1)
    #print('cross_corr_numerator: ', cross_corr_numerator)
    #print(cross_corr_numerator.size())



    """
    JCC_cov =  cross_corr_numerator / sum_cross_corr
    
            = [ [ E[x1'x1] / E[x1'x1] + E[x1'x1],    E[x2'x1] / E[x1'x1] + E[x2'x2],    E[y1'x1] / E[x1'x1] + E[y1'y1],    E[y2'x1] / E[x1'x1] +  E[y2'y2] ], 
                 -------------------------------
         
                [ E[x1'x2] / [E[x2'x2] + E[x1'x1],   E[x2'x2] / E[x2'x2] + E[x2'x2],    E[y1'x2] / E[x2'x2] + E[y1'y1],    E[y2'x2] / E[x2'x2] +  E[y2'y2] ],   
                                                     ------------------------------  
                        
                [ E[x1'y1] / E[y1'y1] + E[x1'x1],    E[x2'y1] / E[y1'y1] + E[x2'x2],    E[y1'y1] / E[y1'y1] + E[y1'y1],    E[y2'y1] / E[y1'y1] +  E[y2'y2] ],  
                                                                                        ------------------------------       
                       
                [ E[x1'y2] / E[y2'y2] + E[x1'x1],    E[x2'y2] / E[y2'y2] + E[x2'x2],    E[y1'y2] / E[y2'y2] + E[y1'y1],    E[y2'y2] / E[y2'y2] +  E[y2'y2] ] ] 
                                                                                                                           -------------------------------
                                                                                                                                                                                                                    
                                                                                                                  
    """

    JCC_cov = cross_corr_numerator / sum_cross_corr
    #print(JCC_cov)


    #print('JCC_cov: ', JCC_cov)

    # follows the numerator () formula for the contrastive loss
    sim = torch.exp(-2 * JCC_cov / temperature)
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

    #print('mask: ',mask)

    #x = sim.masked_select(mask)
    #print('x: ',x)

    #x1 = sim.masked_select(mask).view(n_samples, -1)
    #print('x1: ', x1)

    # Returns a new 1-D tensor which indexes the input tensor (sim) according to the boolean mask (mask) which is a BoolTensor.
    # returns a tensor with 1 row and n columns and sum it with the last dimension
    # masked_select(mask) --> only selects the 'True' value with respect to the mask and the similarity vector
    # .view(n_samples, -1) --> view the unmasked values into the rows with regards to the number of samples and whatever columns that fits
    # .sum(dim=1) --> sum the values in each row, which refers to the negative value for each sample, dim=0 is the outer array
    # neg = [x1 dot product other vectors, x2 dot product other vectors, y1 dot product other vectors, y2 dot product other vectors]
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=1)

    #print('neg: ', neg)

    # Positive similarity
    # exp --> exponential of the sum of the last dimension after output1 * output2 divided by the temp
    """
    JCC loss formula --> (exp(-2 * (E[z_transpose * t])/(E[z_transpose * z] + E[t_transpose * t]))).mean()
    The mean is calculated at the end.
    """
    # reshape to create an extra dimension e.g. (5,3,1)
    # 5 is the batch size
    # 3 is the feature dimension
    # 1 is used to create a d*d cross correlation matrix per batch where the dim1 and dim2 is being transposed
    # e.g. transpose: (5,1,3) * original: (5,3,1) = (5,3,3)
    output1_reshape = torch.reshape(output1, (output1.shape[0],output1.shape[1], 1))
    output2_reshape = torch.reshape(output2, (output2.shape[0], output2.shape[1], 1))

    # shape would be (5,1,3)
    # transpose the feature dimension and keep the batch size the same -> switch the dim1 and dim2 position
    # this will allow a (5,3,3) output after output1_transpose multiply by output1
    # 5 is the batch and 3,3 is the d*d cross correlation matrix
    output1_reshape_transpose = output1_reshape.permute(0,2,1)
    output2_reshape_transpose = output2_reshape.permute(0,2,1)

    #print(output1_reshape_transpose.size())

    # using .bmm as batch matrix multiplication
    # https://pytorch.org/docs/stable/generated/torch.bmm.html
    # returns as (batch_size, 1,1), where the 1 is the sum of the d*d matrix, which is also the cross correlation and expectation
    expectation_output1_output2 = output1_reshape_transpose.bmm(output2_reshape)

    expectation_output1_output1 = output1_reshape_transpose.bmm(output1_reshape)
    #print('expectation_output1_output1: ', expectation_output1_output1)

    expectation_output2_output2 = output2_reshape_transpose.bmm(output2_reshape)

    # get the denominator for the JCC loss
    sumv = expectation_output1_output1 + expectation_output2_output2
    #print('sumv: ', sumv)
    #print(sumv.size())

    mix = expectation_output1_output2 / sumv
    # formula for the JCC loss
    pos = torch.exp((-2 * mix)/temperature)
    #print(pos)

    # concatenate via the rows, stacking vertically
    # there are 2 copies of the positive in 2 different denominators because the loss is symmetric
    # concatenate to get the symmetric loss function with respect to the negative value
    # e.g. neg[0] will be for x1, neg[2] will be for y1, neg[1] will be for x2, neg[3] will be for y2
    # the mean value will be the average loss for which the positions of the images are interchanged.
    # pos = [x1y1, x2y2, x1y1, x2y2]
    pos = torch.cat([pos, pos], dim=0)



    # 2 copies of the numerator as the loss is symmetric but the denominator is 2 different values --> 1 for x, 1 for y
    # the loss will be a scalar value
    loss = -torch.log(pos / neg).mean()
    print(loss)

    #return loss


# batch 4 with 128 features
x1 = torch.randn(4, 128)
#print(y)
#x1 = torch.tensor([[1,2,3,4], [3,4,5,6]])
#x1 = torch.tensor([[1,2], [3,4]])
#x1 = torch.tensor([[1,2]])
#print(x1)
#print(x1.size())


# batch 4 with 128 features
x2 = torch.randn(4, 128)
#x2 = torch.tensor([[5,6,7,8], [7,8,9,10]])
#x2 = torch.tensor([[5,6], [7,8]])
#x2 = torch.tensor([[5,6]])

JCC_contrastive_loss(x1, x2, temperature=0.5)