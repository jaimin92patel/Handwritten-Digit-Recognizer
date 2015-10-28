import numpy as np
import scipy.io as so
import scipy.optimize as opt
import math

def costFunction(nn_params, *args):
    input_layer_size, hidden_layer_size, num_labels, X, Y, lambd = args[0], args[1], args[2], args[3], args[4], args[5]
    length1 = (input_layer_size+1)*(hidden_layer_size)
  
    nn1 = nn_params[:length1]
    T1 = nn1.reshape((hidden_layer_size, input_layer_size+1))
    nn2 = nn_params[length1:]
    T2 = nn2.reshape((num_labels, 1+ hidden_layer_size))      
    
    m = X.shape[0]# number of training examples, useful for calculations
       
    J = 0
    Theta1_grad = np.zeros(T1.shape)
    Theta2_grad = np.zeros(T2.shape)
    D1 = np.zeros(T1.shape)
    D2 = np.zeros(T2.shape)

    for i in range(m):

        #forward prop
        a1 = X[i,:]
        a1 = np.append([1],a1) #Add 1 to front
        z2 = T1.dot(a1) #theta1 * a1
        a2 = sigmoid(z2) #g(z2)
        a2 = np.append([1],a2) #adding 1
        
        z3 = T2.dot(a2)
        a3 = sigmoid(z3)
        forwardH = a3 #value of h(x)
        
        yi = np.zeros((num_labels)) #making changes to Y matrix, converting it to arrays represented by 1's 
        change = Y[i,0]
        if change == 10:
            change = 9
        else:
            change = change - 1
        if change < num_labels:
            yi[change] = 1
        
        J += (-yi.dot(np.log(forwardH)).T - (1-yi).dot(np.log(1-forwardH)).T) #find value of costfunction from h(x) value
        
        #backward prop to find gradients
        delta3 = a3 - yi #Find delta3
        z2 = sigmoidGradient(z2)#Compute derivative of sigmoid(z2)
        
        
        delta2 = ((T2.T).dot(delta3))[1:] * z2
        
        delta3 = delta3.reshape(delta3.shape[0],1)
        a2 = a2.reshape(a2.shape[0],1)
        D2 = np.add(D2,delta3.dot(a2.T))
        
        delta2 = delta2.reshape(delta2.shape[0],1)
        a1 = a1.reshape(a1.shape[0],1)
        temp = delta2.dot(a1.T);
        D1 = np.add(D1,temp);
    
    #Thetagrad Regularization
    Theta2_grad = D2/m
    Theta1_grad = D1/m
   
    
    J = (J+(lambd*(np.square(T1[:,1:]).sum() + np.square(T2[:,1:]).sum()) / 2))/m #regularization
    

    # unroll gradients and concatenate    
    grad = np.concatenate([Theta1_grad.flatten(), Theta2_grad.flatten()])
    return J, grad  

def gradApprox(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambd):
    epsilon = 0.0001
    
   
    gradientApprox = np.zeros(nn_params.size)
    for i in range(nn_params.shape[0]):
        
        nn_params1 = np.copy(nn_params) 
        nn_params2 = np.copy(nn_params)
  
        nn_params1[i] += epsilon
        nn_params2[i] -= epsilon
        cost_plus = costFunction(nn_params1, input_layer_size, hidden_layer_size, num_labels, X, Y, lambd)[0]     
        cost_minus = costFunction(nn_params2, input_layer_size, hidden_layer_size, num_labels, X, Y, lambd)[0]            
        cost_diff = (cost_plus - cost_minus)/(2*.0001)
     
        gradientApprox[i] = cost_diff    
        
    return gradientApprox

    
def sigmoid(h):
    sigmoid = 0
    sigmoid = 1.0 / (1.0 + np.exp(-h))
    return sigmoid
    

def sigmoidGradient(z):
    sigmoidGrad = 0
    sigmoidGrad = np.multiply(sigmoid(z), 1-sigmoid(z))
    return sigmoidGrad
    

def forwardPropAndAccuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y):
    
    predictions = 0
    percentCorrect = 0
    l1 = (input_layer_size+1)*(hidden_layer_size)
  
    nn1 = nn_params[:l1]
    T1 = nn1.reshape((hidden_layer_size, input_layer_size+1))
    nn2 = nn_params[l1:]
    T2 = nn2.reshape((num_labels, 1+ hidden_layer_size))      
    m = X.shape[0]# number of training examples, useful for calculations
    A = np.zeros((m,10))
    Y2 = np.zeros((m,10))

    for i in range(m):
        #forward prop
        a1 = X[i]
        a1 = np.append([1],a1) #add 1 to front
        z2 = T1.dot(a1) #theta1 * a1
        a2 = sigmoid(z2) #g(z2)
        a2 = np.append([1],a2) #adding 1
        
        z3 = T2.dot(a2)
        a3 = sigmoid(z3)
        forwardH = a3 #value of h(x)
        A[i] = forwardH
        
        yi = np.zeros((num_labels)) #making changes to Y matrix, converting it to arrays represented by 1's 
        change = Y[i,0]
        if change == 10:
            change = 9
        else:
            change = change - 1
        if change < num_labels:
            yi[change] = 1
        Y2[i] = yi
    
    A = np.around(A)
    
    for i in range(m):
        if np.array_equal(A[i],Y2[i]):
            predictions = predictions + 1 
    percentCorrect = (predictions*1.0/m)*100
    
   #make sure you return these correctly 
    return predictions, percentCorrect


def randomInitializeWeights(weights, factor):
 
    W = np.random.random(weights.shape)
    #Normalize so that it spans a range of twice epsilon
    W = W * 2 * factor # applied element wise
    #Shift so that mean is at zero  
    W = W - factor#L_in is the number of input units, L_out is the number of output 
    #Units in layer
    
    return W

# Helper methods
def getCost(nn_params, *args):
    input_layer_size, hidden_layer_size, num_labels, X, Y, lambd = args[0], args[1], args[2], args[3], args[4], args[5]
    cost = costFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambd)[0]
    return cost
   
    

def getGrad(nn_params, *args): 
    input_layer_size, hidden_layer_size, num_labels, X, Y, lambd = args[0], args[1], args[2], args[3], args[4], args[5]
    return costFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambd )[1]



###############################################################################################    
##### Start Program! ##########################################################################
###############################################################################################
    

print "Loading Saved Neural Network Parameters..."

data = so.loadmat('ex4data1.mat')

X = data['X']
Y = data['y']

#previously determined weights to check
weights = so.loadmat('ex4weights.mat')

weights1 = weights['Theta1']
weights2 = weights['Theta2']

#weights1 = weights1.T


input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10
lambd = 0
params = np.concatenate([weights1.flatten(), weights2.flatten()])

j, grad = costFunction(params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambd)

print "Cost at parameters loaded from ex4weights.mat. (This value should be about 0.383770 with regularization, 0.287629 without.): ", j

print "signmoidGrad of 0 (should be 0.25): ", sigmoidGradient(0)

params_check = randomInitializeWeights(np.zeros(params.shape), 15)

grad_check = costFunction(params_check[:35], 4, 4, 3, X[:10, :4], Y[:10, :3], lambd)[1]

grad_approx =  gradApprox(params_check[:35], 4, 4, 3, X[:10, :4], Y[:10, :3], lambd)


checkGradient = np.column_stack((grad_check, grad_approx))
print "Gradient check: the two columns should be very close: "
print checkGradient

nn_params = randomInitializeWeights(np.zeros(params.shape), .12)

args = (input_layer_size, hidden_layer_size, num_labels, X, Y, lambd)

result = opt.fmin_cg(getCost, nn_params, fprime=getGrad, args = args, maxiter = 70)

print "Accuracy: ", forwardPropAndAccuracy(result, input_layer_size, hidden_layer_size, num_labels, X, Y)[1]