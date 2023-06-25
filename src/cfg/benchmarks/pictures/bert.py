digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="import numpy as np
import math
def balance_random_3d(depth, l, wid):...
def balance_random_2d(l, wid):...
def arr_add(dst, src):...
def reLU(img):...
def get_mean(row):...
def std_dev(row):...
def BN_layer(arr, weights, biases):...
def fc_layer(arr, W, W_0):...
def softmax(arr):...
def concat(emb, head, tokens, d_k, cur):...
def self_attn(head, tokens, d_k, Q, K, V):...
def main():...
if __name__ == '__main__':
"]
	156 [label="main()
"]
	"156_calls" [label=main shape=box]
	156 -> "156_calls" [label=calls style=dashed]
	1 -> 156 [label="__name__ == '__main__'"]
	subgraph clusterbalance_random_3d {
		graph [label=balance_random_3d]
		3 [label="arr = np.random.rand(depth, l, wid)
neg = True
"]
		"3_calls" [label="np.random.rand" shape=box]
		3 -> "3_calls" [label=calls style=dashed]
		4 [label="for i in range(depth):
"]
		5 [label="for j in range(l):
"]
		7 [label="for k in range(wid):
"]
		9 [label="if neg:
"]
		11 [label="arr[i][j][k] *= -1
"]
		12 [label="neg = not neg
"]
		12 -> 7 [label=""]
		11 -> 12 [label=""]
		9 -> 11 [label=neg]
		9 -> 12 [label="(not neg)"]
		7 -> 9 [label="range(wid)"]
		7 -> 5 [label=""]
		5 -> 7 [label="range(l)"]
		5 -> 4 [label=""]
		4 -> 5 [label="range(depth)"]
		6 [label="return arr
"]
		4 -> 6 [label=""]
		3 -> 4 [label=""]
	}
	subgraph clusterbalance_random_2d {
		graph [label=balance_random_2d]
		16 [label="arr = np.random.rand(l, wid)
neg = True
"]
		"16_calls" [label="np.random.rand" shape=box]
		16 -> "16_calls" [label=calls style=dashed]
		17 [label="for i in range(l):
"]
		18 [label="for j in range(wid):
"]
		20 [label="if neg:
"]
		22 [label="arr[i][j] *= -1
"]
		23 [label="neg = not neg
"]
		23 -> 18 [label=""]
		22 -> 23 [label=""]
		20 -> 22 [label=neg]
		20 -> 23 [label="(not neg)"]
		18 -> 20 [label="range(wid)"]
		18 -> 17 [label=""]
		17 -> 18 [label="range(l)"]
		19 [label="return arr
"]
		17 -> 19 [label=""]
		16 -> 17 [label=""]
	}
	subgraph clusterarr_add {
		graph [label=arr_add]
		27 [label="for i in range(len(dst)):
"]
		28 [label="for j in range(len(dst[0])):
"]
		30 [label="dst[i][j] += src[i][j]
"]
		30 -> 28 [label=""]
		28 -> 30 [label="range(len(dst[0]))"]
		28 -> 27 [label=""]
		27 -> 28 [label="range(len(dst))"]
		29 [label="return dst
"]
		27 -> 29 [label=""]
	}
	subgraph clusterreLU {
		graph [label=reLU]
		35 [label="for i in range(len(img)):
"]
		36 [label="for j in range(len(img[0])):
"]
		38 [label="img[i][j] = max(img[i][j], 0)
"]
		"38_calls" [label=max shape=box]
		38 -> "38_calls" [label=calls style=dashed]
		38 -> 36 [label=""]
		36 -> 38 [label="range(len(img[0]))"]
		36 -> 35 [label=""]
		35 -> 36 [label="range(len(img))"]
		37 [label="return img
"]
		35 -> 37 [label=""]
	}
	subgraph clusterget_mean {
		graph [label=get_mean]
		43 [label="sum_val = 0
"]
		44 [label="for i in range(len(row)):
"]
		45 [label="sum_val += row[i]
"]
		45 -> 44 [label=""]
		44 -> 45 [label="range(len(row))"]
		46 [label="return sum_val / len(row)
"]
		44 -> 46 [label=""]
		43 -> 44 [label=""]
	}
	subgraph clusterstd_dev {
		graph [label=std_dev]
		50 [label="result = 0
"]
		51 [label="for i in range(len(row)):
"]
		52 [label="diff = row[i] - get_mean(row)
result += diff * diff
"]
		"52_calls" [label=get_mean shape=box]
		52 -> "52_calls" [label=calls style=dashed]
		52 -> 51 [label=""]
		51 -> 52 [label="range(len(row))"]
		53 [label="return math.sqrt(result / len(row))
"]
		51 -> 53 [label=""]
		50 -> 51 [label=""]
	}
	subgraph clusterBN_layer {
		graph [label=BN_layer]
		57 [label="for i in range(len(arr)):
"]
		58 [label="dev = std_dev(arr[i])
mean = get_mean(arr[i])
if dev == 0:
"]
		"58_calls" [label="std_dev
get_mean" shape=box]
		58 -> "58_calls" [label=calls style=dashed]
		60 [label="dev = 1
"]
		61 [label="for j in range(len(arr[0])):
"]
		62 [label="arr[i][j] = weights[i] * ((arr[i][j] - mean) / dev) + biases[i]
"]
		62 -> 61 [label=""]
		61 -> 62 [label="range(len(arr[0]))"]
		61 -> 57 [label=""]
		60 -> 61 [label=""]
		58 -> 60 [label="dev == 0"]
		58 -> 61 [label="(dev != 0)"]
		57 -> 58 [label="range(len(arr))"]
		59 [label="return arr
"]
		57 -> 59 [label=""]
	}
	subgraph clusterfc_layer {
		graph [label=fc_layer]
		67 [label="result = np.zeros(len(W[0]))
"]
		"67_calls" [label="np.zeros" shape=box]
		67 -> "67_calls" [label=calls style=dashed]
		68 [label="for i in range(len(W[0])):
"]
		69 [label="sum_val = W_0[i]
"]
		71 [label="for j in range(len(arr)):
"]
		72 [label="sum_val += arr[j] * W[j][i]
"]
		72 -> 71 [label=""]
		71 -> 72 [label="range(len(arr))"]
		73 [label="result[i] = sum_val
"]
		73 -> 68 [label=""]
		71 -> 73 [label=""]
		69 -> 71 [label=""]
		68 -> 69 [label="range(len(W[0]))"]
		70 [label="return result
"]
		68 -> 70 [label=""]
		67 -> 68 [label=""]
	}
	subgraph clustersoftmax {
		graph [label=softmax]
		77 [label="sum_val = 0
"]
		78 [label="for i in range(len(arr)):
"]
		79 [label="sum_val += math.exp(arr[i])
"]
		"79_calls" [label="math.exp" shape=box]
		79 -> "79_calls" [label=calls style=dashed]
		79 -> 78 [label=""]
		78 -> 79 [label="range(len(arr))"]
		80 [label="result = np.zeros(len(arr))
"]
		"80_calls" [label="np.zeros" shape=box]
		80 -> "80_calls" [label=calls style=dashed]
		81 [label="for i in range(len(arr)):
"]
		82 [label="result[i] = math.exp(arr[i]) / sum_val
"]
		"82_calls" [label="math.exp" shape=box]
		82 -> "82_calls" [label=calls style=dashed]
		82 -> 81 [label=""]
		81 -> 82 [label="range(len(arr))"]
		83 [label="return result
"]
		81 -> 83 [label=""]
		80 -> 81 [label=""]
		78 -> 80 [label=""]
		77 -> 78 [label=""]
	}
	subgraph clusterconcat {
		graph [label=concat]
		87 [label="for i in range(tokens):
"]
		88 [label="for j in range(d_k):
"]
		90 [label="emb[i][j + head * d_k] = cur[i][j]
"]
		90 -> 88 [label=""]
		88 -> 90 [label="range(d_k)"]
		88 -> 87 [label=""]
		87 -> 88 [label="range(tokens)"]
		89 [label="return emb
"]
		87 -> 89 [label=""]
	}
	subgraph clusterself_attn {
		graph [label=self_attn]
		95 [label="scores = np.zeros((tokens, tokens))
"]
		"95_calls" [label="np.zeros" shape=box]
		95 -> "95_calls" [label=calls style=dashed]
		96 [label="for i in range(tokens):
"]
		97 [label="for j in range(tokens):
"]
		99 [label="sum_val = 0
"]
		101 [label="for k in range(d_k):
"]
		102 [label="sum_val += Q[head][i][k] * K[head][j][k]
"]
		102 -> 101 [label=""]
		101 -> 102 [label="range(d_k)"]
		103 [label="val = math.sqrt(d_k)
if val == 0:
"]
		"103_calls" [label="math.sqrt" shape=box]
		103 -> "103_calls" [label=calls style=dashed]
		104 [label="val = 1
"]
		105 [label="scores[i][j] = sum_val / val
"]
		105 -> 97 [label=""]
		104 -> 105 [label=""]
		103 -> 104 [label="val == 0"]
		103 -> 105 [label="(val != 0)"]
		101 -> 103 [label=""]
		99 -> 101 [label=""]
		97 -> 99 [label="range(tokens)"]
		100 [label="scores = np.random.rand(tokens, tokens)
scores[i] = softmax(scores[i])
"]
		"100_calls" [label="np.random.rand
softmax" shape=box]
		100 -> "100_calls" [label=calls style=dashed]
		100 -> 96 [label=""]
		97 -> 100 [label=""]
		96 -> 97 [label="range(tokens)"]
		98 [label="out = np.zeros((tokens, d_k))
"]
		"98_calls" [label="np.zeros" shape=box]
		98 -> "98_calls" [label=calls style=dashed]
		106 [label="for i in range(tokens):
"]
		107 [label="for j in range(d_k):
"]
		109 [label="sum_val = 0
"]
		111 [label="for k in range(tokens):
"]
		112 [label="sum_val += scores[i][k] * V[head][k][j]
"]
		112 -> 111 [label=""]
		111 -> 112 [label="range(tokens)"]
		113 [label="out[i][j] = sum_val
"]
		113 -> 107 [label=""]
		111 -> 113 [label=""]
		109 -> 111 [label=""]
		107 -> 109 [label="range(d_k)"]
		107 -> 106 [label=""]
		106 -> 107 [label="range(tokens)"]
		108 [label="return out
"]
		106 -> 108 [label=""]
		98 -> 106 [label=""]
		96 -> 98 [label=""]
		95 -> 96 [label=""]
	}
	subgraph clustermain {
		graph [label=main]
		117 [label="d_model, heads, tokens, layers = 12, 12, 8, 12
d_k = d_model // heads
embeddings = np.random.rand(tokens, d_model)
"]
		"117_calls" [label="np.random.rand" shape=box]
		117 -> "117_calls" [label=calls style=dashed]
		118 [label="for i in range(tokens):
"]
		119 [label="for j in range(d_model):
"]
		121 [label="if j % 2 == 0:
"]
		123 [label="embeddings[i][j] += math.sin(i / math.pow(10000, 2 * j / d_model))
"]
		"123_calls" [label="math.sin" shape=box]
		123 -> "123_calls" [label=calls style=dashed]
		123 -> 119 [label=""]
		121 -> 123 [label="j % 2 == 0"]
		125 [label="embeddings[i][j] += math.cos(i / math.pow(10000, 2 * j / d_model))
"]
		"125_calls" [label="math.cos" shape=box]
		125 -> "125_calls" [label=calls style=dashed]
		125 -> 119 [label=""]
		121 -> 125 [label="(j % 2 != 0)"]
		119 -> 121 [label="range(d_model)"]
		119 -> 118 [label=""]
		118 -> 119 [label="range(tokens)"]
		120 [label="W_Q = balance_random_3d(heads, d_model, d_k)
W_K = balance_random_3d(heads, d_model, d_k)
W_V = balance_random_3d(heads, d_model, d_k)
Q = np.zeros((heads, tokens, d_k))
K = np.zeros((heads, tokens, d_k))
V = np.zeros((heads, tokens, d_k))
"]
		"120_calls" [label="balance_random_3d
balance_random_3d
balance_random_3d
np.zeros
np.zeros
np.zeros" shape=box]
		120 -> "120_calls" [label=calls style=dashed]
		126 [label="for i in range(heads):
"]
		127 [label="for j in range(tokens):
"]
		129 [label="for k in range(d_k):
"]
		131 [label="sumQ, sumK, sumV = 0, 0, 0
"]
		133 [label="for a in range(d_model):
"]
		134 [label="sumQ += embeddings[j][a] * W_Q[i][a][k]
sumK += embeddings[j][a] * W_K[i][a][k]
sumV += embeddings[j][a] * W_V[i][a][k]
"]
		134 -> 133 [label=""]
		133 -> 134 [label="range(d_model)"]
		135 [label="Q[i][j][k] = sumQ
K[i][j][k] = sumK
V[i][j][k] = sumV
"]
		135 -> 129 [label=""]
		133 -> 135 [label=""]
		131 -> 133 [label=""]
		129 -> 131 [label="range(d_k)"]
		129 -> 127 [label=""]
		127 -> 129 [label="range(tokens)"]
		127 -> 126 [label=""]
		126 -> 127 [label="range(heads)"]
		128 [label="for i in range(layers):
"]
		136 [label="emb_cpy = np.copy(embeddings)
multi_head_out = np.zeros((tokens, d_model))
"]
		"136_calls" [label="np.copy
np.zeros" shape=box]
		136 -> "136_calls" [label=calls style=dashed]
		138 [label="for j in range(heads):
"]
		139 [label="cur = self_attn(j, tokens, d_k, Q, K, V)
multi_head_out = concat(multi_head_out, j, tokens, d_k, cur)
"]
		"139_calls" [label="self_attn
concat" shape=box]
		139 -> "139_calls" [label=calls style=dashed]
		139 -> 138 [label=""]
		138 -> 139 [label="range(heads)"]
		140 [label="W_attn = np.random.rand(d_model, d_model)
"]
		"140_calls" [label="np.random.rand" shape=box]
		140 -> "140_calls" [label=calls style=dashed]
		141 [label="for i in range(tokens):
"]
		142 [label="for j in range(d_model):
"]
		144 [label="sum_val = 0
"]
		146 [label="for k in range(d_model):
"]
		147 [label="sum_val += multi_head_out[i][k] * W_attn[k][j]
"]
		147 -> 146 [label=""]
		146 -> 147 [label="range(d_model)"]
		148 [label="embeddings[i][j] = sum_val
"]
		148 -> 142 [label=""]
		146 -> 148 [label=""]
		144 -> 146 [label=""]
		142 -> 144 [label="range(d_model)"]
		142 -> 141 [label=""]
		141 -> 142 [label="range(tokens)"]
		143 [label="embeddings = arr_add(embeddings, emb_cpy)
weights, biases = np.random.rand(d_model), np.random.rand(d_model)
embeddings = BN_layer(embeddings, weights, biases)
emb_cpy = np.copy(embeddings)
W = np.random.rand(d_model, d_model * 4)
W_0 = np.random.rand(d_model * 4)
emb_new = np.zeros((tokens, d_model * 4))
"]
		"143_calls" [label="arr_add
np.random.rand
np.random.rand
BN_layer
np.copy
np.random.rand
np.random.rand
np.zeros" shape=box]
		143 -> "143_calls" [label=calls style=dashed]
		149 [label="for i in range(tokens):
"]
		150 [label="emb_new[i] = fc_layer(embeddings[i], W, W_0)
"]
		"150_calls" [label=fc_layer shape=box]
		150 -> "150_calls" [label=calls style=dashed]
		150 -> 149 [label=""]
		149 -> 150 [label="range(tokens)"]
		151 [label="embeddings = emb_new
embeddings = reLU(embeddings)
W = np.random.rand(d_model * 4, d_model)
W_0 = np.random.rand(d_model)
emb_new = np.zeros((tokens, d_model))
"]
		"151_calls" [label="reLU
np.random.rand
np.random.rand
np.zeros" shape=box]
		151 -> "151_calls" [label=calls style=dashed]
		152 [label="for i in range(tokens):
"]
		153 [label="emb_new[i] = fc_layer(embeddings[i], W, W_0)
"]
		"153_calls" [label=fc_layer shape=box]
		153 -> "153_calls" [label=calls style=dashed]
		153 -> 152 [label=""]
		152 -> 153 [label="range(tokens)"]
		154 [label="embeddings = emb_new
embeddings = arr_add(embeddings, emb_cpy)
embeddings = BN_layer(embeddings, weights, biases)
"]
		"154_calls" [label="arr_add
BN_layer" shape=box]
		154 -> "154_calls" [label=calls style=dashed]
		154 -> 128 [label=""]
		152 -> 154 [label=""]
		151 -> 152 [label=""]
		149 -> 151 [label=""]
		143 -> 149 [label=""]
		141 -> 143 [label=""]
		140 -> 141 [label=""]
		138 -> 140 [label=""]
		136 -> 138 [label=""]
		128 -> 136 [label="range(layers)"]
		126 -> 128 [label=""]
		120 -> 126 [label=""]
		118 -> 120 [label=""]
		117 -> 118 [label=""]
	}
}
