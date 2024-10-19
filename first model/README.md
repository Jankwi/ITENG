# The first model (C++)

This is the first Neural Network I've ever built. I did it in the language which I knew the best -  ```C++```  

**Disclaimer: The code is an iteratively improved mess I created in high school, and should not be treated as a sample of my coding style**

## Abstract
The model is a simple neural network, stored as a computational graph using custom ```C++``` structs. Everything is implemented from scratch - including the forward and backward pass.
The parameters are stored as floats in ```weights.txt``` and ```biases.txt```.

I improved the code iteratively, incorporating multithreading, dynamic learning rate, and other complex techinques. Despite all that the model achieves a mere 65% accuracy on the testing set, and should be treated as a museum exhibit.  
I will not document the code thoroughly, but I will try to give a quick overview of how it works

## Initialization 
```C++
import_everything();

set_current_loss(1);
lowest_loss = loss;
loss_table[0].first = loss;
set_current_loss(0);
loss_table[0].second = loss;

set_step_size(first_step_size);
```
The testing and training set are imported using ```import_everything```

The initial loss for both sets is calculated using  ```set_current_loss()```, and then stored in the ```loss_table```

Since the training loop uses a dynamic step size it first has to be set for all parameters using ```set_step_size()```

## Training Loop
```C++
for (int i = 1; i <=100*1000; i++)
{
	epoch();
	loss_table[i].first = loss;
	set_current_loss(0);
	loss_table[i].second = loss;
	if (i % 50 == 0) update_loss_diary(i);
	if (i % 20 == 0) save_everything();
} 
```
* ```epoch()``` - performs one iteration of gradient descent for all training set examples
* ```set_current_loss(0)``` - calculates the current testing loss
* ```update_loss_diary``` - prints the loss to the log file every 50 epochs
* ```save_setting()``` - saves the parameters responsible for generating the lowest training loss so far

## The rest
I will not be documenting the rest of the code, but will instead just post my favourite part ```#Manualbackpropagation```
```C++
void backward_pass(int index)
{
	ld tangradout = 1 - neurons[index][l+1][1].output * neurons[index][l+1][1].output;
	neurons[index][l + 1][1].grad = 2 * (neurons[index][l+1][1].output - goal[index]);
	neurons[index][l+1][1].grad *= tangradout;
	bias[l+1][1].sum += neurons[index][l+1][1].grad;
	for (int lay = l; lay >= 0; lay--)
	{
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= n; j++) weights[index][lay][i][j].grad = neurons[index][lay][i].output * neurons[index][lay + 1][j].grad;
			for (int j = 1; j <=n; j++) wei[lay][i][j].sum += weights[index][lay][i][j].grad;
		}
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= n; j++) neurons[index][lay][i].grad += wei[lay][i][j].val * neurons[index][lay + 1][j].grad;
			ld tangrad = 1 - neurons[index][lay][i].output * neurons[index][lay][i].output;
			neurons[index][lay][i].grad *= tangrad;
			bias[lay][i].sum += neurons[index][lay][i].grad;
		}
	}
```
