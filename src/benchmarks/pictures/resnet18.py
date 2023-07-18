digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="import math
import numpy as np
from loop import loop
def zero_pad_arr(img, zero_pad):...
def conv_layer(img, filt, numFilt, zero_pad, stride):...
def max_pool(input, l, w, zero_pad, stride):...
def avg_pool(input, l, w, zero_pad, stride):...
def reLU(img):...
def get_mean(row):...
def std_dev(row):...
def BN_layer(img, channels, weights, biases):...
def flatten_layer(img):...
def fc_layer(arr, W, W_0):...
def softmax(img):...
def main():...
if __name__ == '__main__':
"]
	160 [label="loop.start_unroll
main()
"]
	"160_calls" [label=main shape=box]
	160 -> "160_calls" [label=calls style=dashed]
	1 -> 160 [label="__name__ == '__main__'"]
	subgraph clusterzero_pad_arr {
		graph [label=zero_pad_arr]
		3 [label="len_new = len(img[0]) + 2 * zero_pad
wid_new = len(img[0][0]) + 2 * zero_pad
new_img = np.zeros((len(img), len_new, wid_new))
"]
		"3_calls" [label="len
len
np.zeros" shape=box]
		3 -> "3_calls" [label=calls style=dashed]
		4 [label="for i in range(len(img)):
"]
		5 [label="for j in range(len_new):
"]
		7 [label="make_zero = j < zero_pad or j >= len_new - zero_pad
"]
		9 [label="for k in range(wid_new):
"]
		10 [label="if k < zero_pad or k >= wid_new - zero_pad or make_zero:
"]
		12 [label="new_img[i][j][k] = 0
"]
		12 -> 9 [label=""]
		10 -> 12 [label="k < zero_pad or k >= wid_new - zero_pad or make_zero"]
		14 [label="new_img[i][j][k] = img[i][j - zero_pad][k - zero_pad]
"]
		14 -> 9 [label=""]
		10 -> 14 [label="(not (k < zero_pad or k >= wid_new - zero_pad or make_zero))"]
		9 -> 10 [label="range(wid_new)"]
		9 -> 5 [label=""]
		7 -> 9 [label=""]
		5 -> 7 [label="range(len_new)"]
		5 -> 4 [label=""]
		4 -> 5 [label="range(len(img))"]
		6 [label="return new_img
"]
		4 -> 6 [label=""]
		3 -> 4 [label=""]
	}
	subgraph clusterconv_layer {
		graph [label=conv_layer]
		18 [label="f_len = int((len(img[0]) - len(filt[0]) + 2 * zero_pad) / stride + 1)
f_wid = int((len(img[0][0]) - len(filt[0][0]) + 2 * zero_pad) / stride + 1)
biases = np.random.rand(f_len)
img_new = zero_pad_arr(img, zero_pad)
f_new = np.zeros((numFilt, f_len, f_wid))
"]
		"18_calls" [label="int
int
np.random.rand
zero_pad_arr
np.zeros" shape=box]
		18 -> "18_calls" [label=calls style=dashed]
		19 [label="for i in range(numFilt):
"]
		20 [label="for j in range(f_len):
"]
		22 [label="for k in range(f_wid):
"]
		24 [label="for l in range(len(filt)):
"]
		26 [label="for c in range(len(filt[0])):
"]
		28 [label="for d in range(len(filt[0][0])):
"]
		30 [label="f_new[i][j][k] += img_new[l][j * stride + c][k * stride + d] * filt[l][c][d]
"]
		30 -> 28 [label=""]
		28 -> 30 [label="range(len(filt[0][0]))"]
		28 -> 26 [label=""]
		26 -> 28 [label="range(len(filt[0]))"]
		26 -> 24 [label=""]
		24 -> 26 [label="range(len(filt))"]
		24 -> 22 [label=""]
		22 -> 24 [label="range(f_wid)"]
		22 -> 20 [label=""]
		20 -> 22 [label="range(f_len)"]
		20 -> 19 [label=""]
		19 -> 20 [label="range(numFilt)"]
		21 [label="for i in range(numFilt):
"]
		32 [label="for j in range(f_len):
"]
		34 [label="for k in range(f_wid):
"]
		36 [label="f_new[i][j][k] += biases[j]
"]
		36 -> 34 [label=""]
		34 -> 36 [label="range(f_wid)"]
		34 -> 32 [label=""]
		32 -> 34 [label="range(f_len)"]
		32 -> 21 [label=""]
		21 -> 32 [label="range(numFilt)"]
		33 [label="return f_new
"]
		21 -> 33 [label=""]
		19 -> 21 [label=""]
		18 -> 19 [label=""]
	}
	subgraph clustermax_pool {
		graph [label=max_pool]
		41 [label="if zero_pad > 0:
"]
		42 [label="input = zero_pad_arr(input, zero_pad)
"]
		"42_calls" [label=zero_pad_arr shape=box]
		42 -> "42_calls" [label=calls style=dashed]
		43 [label="res_l = int((len(input[0]) - l) / stride + 1)
res_w = int((len(input[0][0]) - w) / stride + 1)
result = np.zeros((len(input), res_l, res_w))
"]
		"43_calls" [label="int
int
np.zeros" shape=box]
		43 -> "43_calls" [label=calls style=dashed]
		44 [label="for i in range(len(input)):
"]
		45 [label="for j in range(res_l):
"]
		47 [label="for k in range(res_w):
"]
		49 [label="for c in range(l):
"]
		51 [label="for d in range(w):
"]
		53 [label="result[i][j][k] = max(input[i][j][k], result[i][j][k], input[i][j * stride +
    c][k * stride + d])
"]
		"53_calls" [label=max shape=box]
		53 -> "53_calls" [label=calls style=dashed]
		53 -> 51 [label=""]
		51 -> 53 [label="range(w)"]
		51 -> 49 [label=""]
		49 -> 51 [label="range(l)"]
		49 -> 47 [label=""]
		47 -> 49 [label="range(res_w)"]
		47 -> 45 [label=""]
		45 -> 47 [label="range(res_l)"]
		45 -> 44 [label=""]
		44 -> 45 [label="range(len(input))"]
		46 [label="for i in range(len(input)):
"]
		55 [label="for j in range(res_l):
"]
		57 [label="for k in range(res_w):
"]
		59 [label="result[i][j][k] /= l * w
"]
		59 -> 57 [label=""]
		57 -> 59 [label="range(res_w)"]
		57 -> 55 [label=""]
		55 -> 57 [label="range(res_l)"]
		55 -> 46 [label=""]
		46 -> 55 [label="range(len(input))"]
		56 [label="return result
"]
		46 -> 56 [label=""]
		44 -> 46 [label=""]
		43 -> 44 [label=""]
		42 -> 43 [label=""]
		41 -> 42 [label="zero_pad > 0"]
		41 -> 43 [label="(zero_pad <= 0)"]
	}
	subgraph clusteravg_pool {
		graph [label=avg_pool]
		64 [label="if zero_pad > 0:
"]
		65 [label="input = zero_pad_arr(input, zero_pad)
"]
		"65_calls" [label=zero_pad_arr shape=box]
		65 -> "65_calls" [label=calls style=dashed]
		66 [label="res_l = int((len(input[0]) - l) / stride + 1)
res_w = int((len(input[0][0]) - w) / stride + 1)
result = np.zeros((len(input), res_l, res_w))
"]
		"66_calls" [label="int
int
np.zeros" shape=box]
		66 -> "66_calls" [label=calls style=dashed]
		67 [label="for i in range(len(input)):
"]
		68 [label="for j in range(res_l):
"]
		70 [label="for k in range(res_w):
"]
		72 [label="for c in range(l):
"]
		74 [label="for d in range(w):
"]
		76 [label="result[i][j][k] += input[i][j * stride + c][k * stride + d]
"]
		76 -> 74 [label=""]
		74 -> 76 [label="range(w)"]
		74 -> 72 [label=""]
		72 -> 74 [label="range(l)"]
		72 -> 70 [label=""]
		70 -> 72 [label="range(res_w)"]
		70 -> 68 [label=""]
		68 -> 70 [label="range(res_l)"]
		68 -> 67 [label=""]
		67 -> 68 [label="range(len(input))"]
		69 [label="for i in range(len(input)):
"]
		78 [label="for j in range(res_l):
"]
		80 [label="for k in range(res_w):
"]
		82 [label="result[i][j][k] /= l * w
"]
		82 -> 80 [label=""]
		80 -> 82 [label="range(res_w)"]
		80 -> 78 [label=""]
		78 -> 80 [label="range(res_l)"]
		78 -> 69 [label=""]
		69 -> 78 [label="range(len(input))"]
		79 [label="return result
"]
		69 -> 79 [label=""]
		67 -> 69 [label=""]
		66 -> 67 [label=""]
		65 -> 66 [label=""]
		64 -> 65 [label="zero_pad > 0"]
		64 -> 66 [label="(zero_pad <= 0)"]
	}
	subgraph clusterreLU {
		graph [label=reLU]
		87 [label="for i in range(len(img)):
"]
		88 [label="for j in range(len(img[0])):
"]
		90 [label="for k in range(len(img[0][0])):
"]
		92 [label="img[i][j][k] = max(img[i][j][k], 0)
"]
		"92_calls" [label=max shape=box]
		92 -> "92_calls" [label=calls style=dashed]
		92 -> 90 [label=""]
		90 -> 92 [label="range(len(img[0][0]))"]
		90 -> 88 [label=""]
		88 -> 90 [label="range(len(img[0]))"]
		88 -> 87 [label=""]
		87 -> 88 [label="range(len(img))"]
		89 [label="return img
"]
		87 -> 89 [label=""]
	}
	subgraph clusterget_mean {
		graph [label=get_mean]
		97 [label="sum_val = 0
"]
		98 [label="for i in range(len(row)):
"]
		99 [label="sum_val += row[i]
"]
		99 -> 98 [label=""]
		98 -> 99 [label="range(len(row))"]
		100 [label="return sum_val / len(row)
"]
		98 -> 100 [label=""]
		97 -> 98 [label=""]
	}
	subgraph clusterstd_dev {
		graph [label=std_dev]
		104 [label="result = 0
"]
		105 [label="for i in range(len(row)):
"]
		106 [label="diff = row[i] - get_mean(row)
result += diff * diff
"]
		"106_calls" [label=get_mean shape=box]
		106 -> "106_calls" [label=calls style=dashed]
		106 -> 105 [label=""]
		105 -> 106 [label="range(len(row))"]
		107 [label="return math.sqrt(result / len(row))
"]
		105 -> 107 [label=""]
		104 -> 105 [label=""]
	}
	subgraph clusterBN_layer {
		graph [label=BN_layer]
		111 [label="for i in range(channels):
"]
		112 [label="for j in range(len(img[0])):
"]
		114 [label="dev = std_dev(img[i][j][:])
mean = get_mean(img[i][j][:])
if dev == 0.0:
"]
		"114_calls" [label="std_dev
get_mean" shape=box]
		114 -> "114_calls" [label=calls style=dashed]
		116 [label="dev = 1.0
"]
		117 [label="for k in range(len(img[0][0])):
"]
		118 [label="img[i][j][k] = weights[j] * ((img[i][j][k] - mean) / dev) + biases[j]
"]
		118 -> 117 [label=""]
		117 -> 118 [label="range(len(img[0][0]))"]
		117 -> 112 [label=""]
		116 -> 117 [label=""]
		114 -> 116 [label="dev == 0.0"]
		114 -> 117 [label="(dev != 0.0)"]
		112 -> 114 [label="range(len(img[0]))"]
		112 -> 111 [label=""]
		111 -> 112 [label="range(channels)"]
		113 [label="return img
"]
		111 -> 113 [label=""]
	}
	subgraph clusterflatten_layer {
		graph [label=flatten_layer]
		123 [label="result = np.zeros(len(img) * len(img[0]) * len(img[0][0]))
index = 0
"]
		"123_calls" [label="np.zeros" shape=box]
		123 -> "123_calls" [label=calls style=dashed]
		124 [label="for i in range(len(img)):
"]
		125 [label="for j in range(len(img[0])):
"]
		127 [label="for k in range(len(img[0][0])):
"]
		129 [label="result[index] = img[i][j][k]
index += 1
"]
		129 -> 127 [label=""]
		127 -> 129 [label="range(len(img[0][0]))"]
		127 -> 125 [label=""]
		125 -> 127 [label="range(len(img[0]))"]
		125 -> 124 [label=""]
		124 -> 125 [label="range(len(img))"]
		126 [label="return result
"]
		124 -> 126 [label=""]
		123 -> 124 [label=""]
	}
	subgraph clusterfc_layer {
		graph [label=fc_layer]
		134 [label="result = np.zeros(len(W[0]))
"]
		"134_calls" [label="np.zeros" shape=box]
		134 -> "134_calls" [label=calls style=dashed]
		135 [label="for i in range(len(W[0])):
"]
		136 [label="sum_val = W_0[i]
"]
		138 [label="for j in range(len(arr)):
"]
		139 [label="sum_val += arr[j] * W[j][i]
"]
		139 -> 138 [label=""]
		138 -> 139 [label="range(len(arr))"]
		140 [label="result[i] = sum_val
"]
		140 -> 135 [label=""]
		138 -> 140 [label=""]
		136 -> 138 [label=""]
		135 -> 136 [label="range(len(W[0]))"]
		137 [label="return result
"]
		135 -> 137 [label=""]
		134 -> 135 [label=""]
	}
	subgraph clustersoftmax {
		graph [label=softmax]
		144 [label="sum_val = 0
"]
		145 [label="for i in range(len(img)):
"]
		146 [label="sum_val += math.exp(img[i])
"]
		"146_calls" [label="math.exp" shape=box]
		146 -> "146_calls" [label=calls style=dashed]
		146 -> 145 [label=""]
		145 -> 146 [label="range(len(img))"]
		147 [label="result = np.zeros(len(img))
"]
		"147_calls" [label="np.zeros" shape=box]
		147 -> "147_calls" [label=calls style=dashed]
		148 [label="for i in range(len(img)):
"]
		149 [label="result[i] = math.exp(img[i]) / sum_val
"]
		"149_calls" [label="math.exp" shape=box]
		149 -> "149_calls" [label=calls style=dashed]
		149 -> 148 [label=""]
		148 -> 149 [label="range(len(img))"]
		150 [label="return result
"]
		148 -> 150 [label=""]
		147 -> 148 [label=""]
		145 -> 147 [label=""]
		144 -> 145 [label=""]
	}
	subgraph clustermain {
		graph [label=main]
		154 [label="zero_pad = 3
stride = 2
filt = np.random.rand(3, 7, 7)
img = np.random.rand(3, 64, 64)
img = conv_layer(img, filt, 1, zero_pad, stride)
weights = np.random.rand(len(img[0]))
biases = np.random.rand(len(img[0]))
img = BN_layer(img, 1, weights, biases)
img = reLU(img)
img = max_pool(img, 3, 3, 1, 2)
filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
"]
		"154_calls" [label="np.random.rand
np.random.rand
conv_layer
np.random.rand
np.random.rand
BN_layer
reLU
max_pool
np.random.rand" shape=box]
		154 -> "154_calls" [label=calls style=dashed]
		155 [label="for i in range(2):
"]
		156 [label="byPass = np.copy(img)
img = conv_layer(img, filt, 1, 3, 1)
img = BN_layer(img, 1, weights, biases)
img = reLU(img)
img = conv_layer(img, filt, 1, 3, 1)
img += byPass
img = reLU(img)
"]
		"156_calls" [label="np.copy
conv_layer
BN_layer
reLU
conv_layer
reLU" shape=box]
		156 -> "156_calls" [label=calls style=dashed]
		156 -> 155 [label=""]
		155 -> 156 [label="range(2)"]
		157 [label="filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
byPass = np.copy(img)
byPass = conv_layer(byPass, filt, 1, 3, 2)
byPass = BN_layer(byPass, 1, weights, biases)
img = conv_layer(img, filt, 1, 3, 1)
img = BN_layer(img, 1, weights, biases)
img = reLU(img)
img = conv_layer(img, filt, 1, 3, 2)
img += byPass
img = reLU(img)
byPass = np.copy(img)
img = conv_layer(img, filt, 1, 3, 1)
img = BN_layer(img, 1, weights, biases)
img = reLU(img)
img = conv_layer(img, filt, 1, 3, 1)
img = BN_layer(img, 1, weights, biases)
img += byPass
img = reLU(img)
filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
byPass = np.copy(img)
byPass = conv_layer(byPass, filt, 1, 3, 2)
byPass = BN_layer(byPass, 1, weights, biases)
img = conv_layer(img, filt, 1, 3, 1)
img = BN_layer(img, 1, weights, biases)
img = reLU(img)
img = conv_layer(img, filt, 1, 3, 2)
img += byPass
img = reLU(img)
byPass = np.copy(img)
img = conv_layer(img, filt, 1, 3, 1)
img = BN_layer(img, 1, weights, biases)
img = reLU(img)
img = conv_layer(img, filt, 1, 3, 1)
img = BN_layer(img, 1, weights, biases)
img += byPass
img = reLU(img)
filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
byPass = np.copy(img)
byPass = conv_layer(byPass, filt, 1, 3, 2)
byPass = BN_layer(byPass, 1, weights, biases)
img = conv_layer(img, filt, 1, 3, 1)
img = BN_layer(img, 1, weights, biases)
img = reLU(img)
img = conv_layer(img, filt, 1, 3, 2)
img += byPass
img = reLU(img)
byPass = np.copy(img)
img = conv_layer(img, filt, 1, 3, 1)
img = BN_layer(img, 1, weights, biases)
img = reLU(img)
img = conv_layer(img, filt, 1, 3, 1)
img = BN_layer(img, 1, weights, biases)
img += byPass
img = reLU(img)
img = avg_pool(img, len(img[0]), len(img[0][0]), 3, 1)
flat = flatten_layer(img)
weights = np.random.rand(len(img) * len(img[0]) * len(img[0][0]), 7)
w_0 = np.random.rand(7)
flat = fc_layer(flat, weights, w_0)
final = softmax(flat)
return 0
"]
		"157_calls" [label="np.random.rand
np.copy
conv_layer
BN_layer
conv_layer
BN_layer
reLU
conv_layer
reLU
np.copy
conv_layer
BN_layer
reLU
conv_layer
BN_layer
reLU
np.random.rand
np.copy
conv_layer
BN_layer
conv_layer
BN_layer
reLU
conv_layer
reLU
np.copy
conv_layer
BN_layer
reLU
conv_layer
BN_layer
reLU
np.random.rand
np.copy
conv_layer
BN_layer
conv_layer
BN_layer
reLU
conv_layer
reLU
np.copy
conv_layer
BN_layer
reLU
conv_layer
BN_layer
reLU
avg_pool
flatten_layer
np.random.rand
np.random.rand
fc_layer
softmax" shape=box]
		157 -> "157_calls" [label=calls style=dashed]
		155 -> 157 [label=""]
		154 -> 155 [label=""]
	}
}
