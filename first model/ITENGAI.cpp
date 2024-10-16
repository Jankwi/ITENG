#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <thread>
//#include <Windows.h>
using namespace std;
#define ld long double

struct node
{
	ld sum, sb, output;
	ld grad;
};
struct edge
{
	ld grad;
};
struct swiss_knife
{
	ld sum; ld prevgrad;
	ld step_size; ld val;
};
const int l = 3; //number of layers
const int L = l + 3; // number of layers + 3
const int n = 400;
const int N = n + 3; //number of neurons in each layer + 3
const int tr = 2000;
const int TR = tr + 3; //training set size + 3
const int te = 2700;
const int TE = te + 3;
const int th = 10;
const int TH = th + 3; //number of threads + 3
const int batch_size_tr = 200;
const int batch_size_te = 270;
const int E = 1e5; // upper_bound (overestimated) for the number of epochs
const int INF = 1e9+3;
int input_tr[TR][N]; int inp_tr[TR]; ld ans_tr[TR];
int input_te[TE][N]; int inp_te[TR]; ld ans_te[TE];
thread* parallel[TH];
node neurons[TH][L][N];
edge weights[TH][L][N][N];
ld lowest_loss = INF;
ld bb[L][N]; ld bw[L][N][N];
/*ld sum_b[L][N]; ld sum_w[L][N][N];
ld prevgrad_b[L][N]; ld prevgrad_w[L][N][N];
ld step_size_b[L][N]; ld step_size_w[L][N][N];
ld bias[L][N]; ld wei[L][N][N]; */
swiss_knife bias[L][N];
swiss_knife wei[L][N][N];
pair<ld, ld> loss_table[E];
ld local_loss[TH]; ld goal[TH];
ld loss = 0;
ld cap = 0.001;
void import_weights()
{
	ifstream inputfile("weights.txt");
	streambuf* orig_cin_buf = std::cin.rdbuf(inputfile.rdbuf());
	for (int lay = 0; lay < l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= n; j++)
			{
				cin >> wei[lay][i][j].val;
				bw[lay][i][j] = wei[lay][i][j].val;
			}
		}
	}
	for (int j = 1; j <= n; j++)
	{
		cin >> wei[l][j][1].val;
		bw[l][j][1] = wei[l][j][1].val;
	}
	cin.rdbuf(orig_cin_buf);
	inputfile.close();
}
void import_biases()
{
	ifstream inputfile("biases.txt");
	streambuf* orig_cin_buf = std::cin.rdbuf(inputfile.rdbuf());
	for (int lay = 1; lay <= l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			cin >> bias[lay][i].val;
			bb[lay][i] = bias[lay][i].val;
		}
	}
	cin >> bias[l + 1][1].val;
	bb[l+1][1] = bias[l + 1][1].val;
	cin.rdbuf(orig_cin_buf);
	inputfile.close();
}
void import_training_set()
{
	ifstream inputfile("training_set.txt");
	streambuf* orig_cin_buf = std::cin.rdbuf(inputfile.rdbuf());
	for (int i = 1; i <=tr; i++)
	{
		string s; cin >> s;
		if (i == 1410) cout << "striiiiiiiiiing  " << s << "\n";
		for (int j = 0; j < s.size(); j++) 
		{
			if('a' <= s[j] && s[j] <= 'z')	input_tr[i][j+1] = s[j]-'a'+1;
			if('A' <= s[j] && s[j] <= 'Z') input_tr[i][j+1] = s[j] - 'A' + 1;
		}
		//cout << "\n";
		inp_tr[i] = s.size();
		cin >> ans_tr[i];

	}
	cin.rdbuf(orig_cin_buf);
	inputfile.close();
}
void import_testing_set()
{
	ifstream inputfile("testing_set.txt");
	streambuf* orig_cin_buf = std::cin.rdbuf(inputfile.rdbuf());
	for (int i = 1; i <= te; i++)
	{
		string s; cin >> s;
		if (i == 2137) cout << "striiiiiiiiiing  " << s << "\n";
		for (int j = 0; j < s.size(); j++)
		{
			if ('a' <= s[j] && s[j] <= 'z')	input_te[i][j + 1] = s[j] - 'a' + 1;
			if ('A' <= s[j] && s[j] <= 'Z') input_te[i][j + 1] = s[j] - 'A' + 1;
		}
		//cout << "\n";
		inp_te[i] = s.size();
		cin >> ans_te[i];
	}
	cin.rdbuf(orig_cin_buf);
	inputfile.close();
}
void import_everything()
{
	import_weights();
	import_biases();
	import_training_set();
	import_testing_set();
}
void save_weights()
{
	std::ofstream outputFile("weights.txt", std::ios::out | std::ios::trunc);

	streambuf* orig_cout_buf = std::cout.rdbuf(outputFile.rdbuf());
	std::ostream& cout = std::cout;
	for (int lay = 0; lay < l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= n; j++)
			{
				cout << bw[lay][i][j] << " ";
			}
			cout << "\n";
		}
	}
	for (int j = 1; j <= n; j++)
	{
		cout << bw[l][j][1] << " ";
	}
	std::cout.rdbuf(orig_cout_buf);
	outputFile.close();
}
void save_biases()
{
	ofstream outputFile("biases.txt", std::ios::out | std::ios::trunc);

	streambuf* orig_cout_buf = std::cout.rdbuf(outputFile.rdbuf());
	for (int lay = 1; lay <= l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			cout << bb[lay][i] << " ";
		}
		cout << "\n";
	}
	cout << bb[l + 1][1];

	std::cout.rdbuf(orig_cout_buf);
	outputFile.close();
}
void save_everything()
{
	save_weights();
	save_biases();
}
void update_loss_diary(int ub)
{
	ofstream outputFile("loss_diary.txt", std::ios::out | std::ios::trunc);

	streambuf* orig_cout_buf = std::cout.rdbuf(outputFile.rdbuf());

	for (int i = 0; i <= ub; i++)
	{

		cout << "epoch " << i << "  training_set_loss " << fixed << setprecision(20) <<  loss_table[i].first << "  testing_set_loss " << loss_table[i].second << "\n";
	}

	std::cout.rdbuf(orig_cout_buf);
	outputFile.close();
}

void forward_pass(int index)
{
	//cout << "forwaaaaard ";
	for (int lay = 1; lay <= l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= n; j++) neurons[index][lay][i].sum += neurons[index][lay - 1][j].output * wei[lay - 1][j][i].val;
			neurons[index][lay][i].sb = neurons[index][lay][i].sum + bias[lay][i].val;
			neurons[index][lay][i].output = tanh(neurons[index][lay][i].sb);
		}
	}
	for (int i = 1; i <= n; i++) neurons[index][l + 1][1].sum += neurons[index][l][i].output * wei[l][i][1].val;
	neurons[index][l + 1][1].sb = neurons[index][l + 1][1].sum + bias[l + 1][1].val;
	neurons[index][l + 1][1].output = tanh(neurons[index][l + 1][1].sb);
	ld result = neurons[index][l + 1][1].output - goal[index]; result *= result;
	local_loss[index] += result;
	//cout << "the output is equal to " << neurons[index][l+1][1].output;
}
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
}
void clear_neurons(int index)
{
	for (int i = 1; i <= n; i++) neurons[index][0][i].output = 0;
	for (int lay = 1; lay <= l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= n; j++) neurons[index][lay][i].sum = 0;
			neurons[index][lay][i].sb = 0;
			neurons[index][lay][i].output = 0;
		}
	}
	for (int i = 1; i <= n; i++) neurons[index][l + 1][1].sum = 0;
	neurons[index][l + 1][1].sb = 0;
	neurons[index][l+1][1].output = 0;
	goal[index] = 0;
}	
void clear_grads(int index)
{
	for (int lay = 0; lay < l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= n; j++)
			{
				weights[index][lay][i][j].grad = 0;
			}
		}
	}
	for (int j = 1; j <= n; j++)
	{
		weights[index][l][j][1].grad = 0;
	}

	for (int lay = 1; lay <= l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			neurons[index][lay][i].grad = 0;
		}
	}
	neurons[index][l + 1][1].grad = 0;
}
void clear_sums()
{
	for (int lay = 0; lay < l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= n; j++)
			{
				wei[lay][i][j].sum = 0;
			}
		}
	}
	for (int j = 1; j <= n; j++)
	{
		wei[l][j][1].sum = 0;
	}

	for (int lay = 1; lay <= l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			bias[lay][i].sum = 0;
		}
	}
	bias[l + 1][1].sum = 0;
}
void add_grads()
{
	for (int lay = 0; lay < l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= n; j++)
			{
				if (wei[lay][i][j].sum * wei[lay][i][j].prevgrad >= 0 && wei[lay][i][j].step_size <= cap) wei[lay][i][j].step_size *= 1.2;
				else wei[lay][i][j].step_size *= 0.5;

				wei[lay][i][j].val -= wei[lay][i][j].step_size * wei[lay][i][j].step_size;

				wei[lay][i][j].prevgrad = wei[lay][i][j].sum;
			}
		}
	}
	for (int j = 1; j <= n; j++)
	{
		if (wei[l][j][1].sum * wei[l][j][1].prevgrad >= 0 && wei[l][j][1].step_size <= cap) wei[l][j][1].step_size *= 1.2;
		else wei[l][j][1].step_size *= 0.5;

		wei[l][j][1].val -= wei[l][j][1].step_size * wei[l][j][1].sum;

		wei[l][j][1].prevgrad = wei[l][j][1].sum;
	}

	for (int lay = 1; lay <= l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			if (bias[lay][i].sum * bias[lay][i].prevgrad >= 0 && bias[lay][i].step_size <= cap) bias[lay][i].step_size *= 1.2;
			else bias[lay][i].step_size *= 0.5;


			bias[lay][i].val -= bias[lay][i].step_size * bias[lay][i].sum;

			bias[lay][i].prevgrad = bias[lay][i].sum;
		}
	}

	if (bias[l+1][1].sum * bias[l+1][1].prevgrad >= 0 && bias[l+1][1].step_size <= cap) bias[l + 1][1].step_size *= 1.2;
	else bias[l + 1][1].step_size *= 0.5;


	bias[l+1][1].val -= bias[l + 1][1].step_size  * bias[l+1][1].sum;

	bias[l+1][1].prevgrad = bias[l+1][1].sum;
}
void update_best()
{
	lowest_loss = loss;
	for (int lay = 0; lay < l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= n; j++)
			{
				bw[lay][i][j] = wei[lay][i][j].val;
			}
		}
	}
	for (int j = 1; j <= n; j++)
	{
		bw[l][j][1] =  wei[l][j][1].val;
	}

	for (int lay = 1; lay <= l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			bb[lay][i] = bias[lay][i].val;
		}
	}
	bb[l+1][1] = bias[l+1][1].val;
}

void run_test(int x, int index, bool training)
{
	//cout << "xdddddd - ";
	//for (int i = 1; i <=n; i++) cout << input[x][i] << "\n";
	for (int i = 0; i < N-2; i++) neurons[index][0][i].output = 0;
	if (training)
	{
		for (int i = 1; i <= inp_tr[x]; i++)
		{
			int id = input_tr[x][i];
			id += (i - 1) * 26;
			neurons[index][0][id].output = 1;
		}
		//cout << "inputs \n";
		//for (int i = 1; i <=10; i++) cout << neurons[0][i].output << " ";
		//cout << "\n";
		goal[index] = ans_tr[x];
		forward_pass(index);
	}
	else
	{
		for (int i = 1; i <= inp_te[x]; i++)
		{
			int id = input_te[x][i];
			id += (i - 1) * 26;
			neurons[index][0][id].output = 1;
		}
		//cout << "inputs \n";
		//for (int i = 1; i <=10; i++) cout << neurons[0][i].output << " ";
		//cout << "\n";
		goal[index] = ans_tr[x];
		forward_pass(index);
	}
}
void single_loss_calc(int x, int index, int lb, int ub, bool training)
{
	clear_neurons(index); clear_grads(index);
	run_test(x, index, training);
}
void single_thread(int index, int lb, int ub, bool if_backward, bool training)
{
	//Sleep(index * 70);
	//cout << "wypisuje\n" << index << " " << lb << " " << ub << "\n";
	for (int i = lb; i <= ub; i++)
	{
		single_loss_calc(i, index, lb, ub, training);
		if (if_backward) backward_pass(index);
	}
}
void epoch()
{
	clear_sums();

	for (int i = 0; i < th; i++)
	{
		parallel[i] = new thread(single_thread, i, i * batch_size_tr + 1, min(tr, (i + 1) * batch_size_tr), 1, 1);
	}
	for (int i = 0; i < th; i++)
	{
		parallel[i]->join();
	}

	loss = 0;
	for (int i = 0; i < th; i++) local_loss[i] = 0;
	add_grads();
	for (int i = 0; i < th; i++)
	{
		parallel[i] = new thread(single_thread, i, i * batch_size_tr +1, min(tr, (i + 1) * batch_size_tr), 0, 1);
	}
	for (int i = 0; i < th; i++)
	{
		parallel[i]->join();
	}

	for (int i = 0; i < th; i++) loss += local_loss[i];
	if(loss <= lowest_loss)
	{
		update_best();
	}
}
void set_current_loss(bool training)
{
	loss = 0;
	for (int i = 0; i < th; i++) local_loss[i] = 0;

	for (int i = 0; i < th; i++)
	{
		//cout << "jestem w " << i <<  "\n";
		if (training) parallel[i] = new thread(single_thread, i, i * batch_size_tr +1, min(tr, (i + 1) * batch_size_tr), 0, training);
		else parallel[i] = new thread(single_thread, i, i * batch_size_te + 1, min(te, (i + 1) * batch_size_te), 0, training);
	}
	for (int i = 0; i < th; i++)
	{
		parallel[i]->join();
	}
	//cout << "SIUUUUUUU again \n";

	for (int i = 0; i < th; i++) loss += local_loss[i];
}
void set_step_size(ld step_size)
{
	for (int lay = 0; lay < l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= n; j++)
			{
				wei[lay][i][j].step_size = step_size;
			}
		}
	}
	for (int j = 1; j <= n; j++)
	{
		wei[l][j][1].step_size = step_size;
	}


	for (int lay = 1; lay <= l; lay++)
	{
		for (int i = 1; i <= n; i++)
		{
			bias[lay][i].step_size = step_size;
		}
	}
	bias[l + 1][1].step_size = step_size;
}

int main()
{
	import_everything();
	cout << "SIUUUUUUUUUU\n";
	set_current_loss(1);
	lowest_loss = loss;
	cout << "epoch 0  ";
	cout << "training_set_loss - " << loss << " ";
	loss_table[0].first = loss;
	set_current_loss(0);
	cout << "testing_set_loss - " << loss << "\n";
	loss_table[0].second = loss;
	ld first_step_size = 0.00003;
	set_step_size(first_step_size);

	for (int i = 1; i <=100*1000; i++)
	{
		cout << "epoch " << i << "   ";
		epoch();
		cout << "training_set_loss - " << fixed << setprecision(7) << loss << " ";
		loss_table[i].first = loss;
		set_current_loss(0);
		cout << "testing_set_loss - " << fixed << setprecision(7) << loss << "\n";
		loss_table[i].second = loss;
		if (i % 50 == 0) update_loss_diary(i);
		if(i % 20 == 0) save_everything();
	} 
	save_everything();
}
