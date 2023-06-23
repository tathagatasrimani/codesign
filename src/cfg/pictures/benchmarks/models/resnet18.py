digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="import math
import numpy as np
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
	145 [label="main()
"]
	"145_calls" [label=main shape=box]
	145 -> "145_calls" [label=calls style=dashed]
	1 -> 145 [label="__name__ == '__main__'"]
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
		24 [label="sum_val = biases[j]
"]
		26 [label="for l in range(len(filt)):
"]
		27 [label="for c in range(len(filt[0])):
"]
		29 [label="for d in range(len(filt[0][0])):
"]
		31 [label="sum_val += img_new[l][j * stride + c][k * stride + d] * filt[l][c][d]
"]
		31 -> 29 [label=""]
		29 -> 31 [label="range(len(filt[0][0]))"]
		29 -> 27 [label=""]
		27 -> 29 [label="range(len(filt[0]))"]
		27 -> 26 [label=""]
		26 -> 27 [label="range(len(filt))"]
		28 [label="f_new[i][j][k] = sum_val
"]
		28 -> 22 [label=""]
		26 -> 28 [label=""]
		24 -> 26 [label=""]
		22 -> 24 [label="range(f_wid)"]
		22 -> 20 [label=""]
		20 -> 22 [label="range(f_len)"]
		20 -> 19 [label=""]
		19 -> 20 [label="range(numFilt)"]
		21 [label="return f_new
"]
		19 -> 21 [label=""]
		18 -> 19 [label=""]
	}
	subgraph clustermax_pool {
		graph [label=max_pool]
		36 [label="if zero_pad > 0:
"]
		37 [label="input = zero_pad_arr(input, zero_pad)
"]
		"37_calls" [label=zero_pad_arr shape=box]
		37 -> "37_calls" [label=calls style=dashed]
		38 [label="res_l = int((len(input[0]) - l) / stride + 1)
res_w = int((len(input[0][0]) - w) / stride + 1)
result = np.zeros((len(input), res_l, res_w))
"]
		"38_calls" [label="int
int
np.zeros" shape=box]
		38 -> "38_calls" [label=calls style=dashed]
		39 [label="for i in range(len(input)):
"]
		40 [label="for j in range(res_l):
"]
		42 [label="for k in range(res_w):
"]
		44 [label="max_val = input[i][j][k]
"]
		46 [label="for c in range(l):
"]
		47 [label="for d in range(w):
"]
		49 [label="max_val = max(max_val, input[i][j * stride + c][k * stride + d])
"]
		"49_calls" [label=max shape=box]
		49 -> "49_calls" [label=calls style=dashed]
		49 -> 47 [label=""]
		47 -> 49 [label="range(w)"]
		47 -> 46 [label=""]
		46 -> 47 [label="range(l)"]
		48 [label="result[i][j][k] = max_val / (l * w)
"]
		48 -> 42 [label=""]
		46 -> 48 [label=""]
		44 -> 46 [label=""]
		42 -> 44 [label="range(res_w)"]
		42 -> 40 [label=""]
		40 -> 42 [label="range(res_l)"]
		40 -> 39 [label=""]
		39 -> 40 [label="range(len(input))"]
		41 [label="return result
"]
		39 -> 41 [label=""]
		38 -> 39 [label=""]
		37 -> 38 [label=""]
		36 -> 37 [label="zero_pad > 0"]
		36 -> 38 [label="(zero_pad <= 0)"]
	}
	subgraph clusteravg_pool {
		graph [label=avg_pool]
		54 [label="if zero_pad > 0:
"]
		55 [label="input = zero_pad_arr(input, zero_pad)
"]
		"55_calls" [label=zero_pad_arr shape=box]
		55 -> "55_calls" [label=calls style=dashed]
		56 [label="res_l = int((len(input[0]) - l) / stride + 1)
res_w = int((len(input[0][0]) - w) / stride + 1)
result = np.zeros((len(input), res_l, res_w))
"]
		"56_calls" [label="int
int
np.zeros" shape=box]
		56 -> "56_calls" [label=calls style=dashed]
		57 [label="for i in range(len(input)):
"]
		58 [label="for j in range(res_l):
"]
		60 [label="for k in range(res_w):
"]
		62 [label="sum_val = 0
"]
		64 [label="for c in range(l):
"]
		65 [label="for d in range(w):
"]
		67 [label="sum_val += input[i][j * stride + c][k * stride + d]
"]
		67 -> 65 [label=""]
		65 -> 67 [label="range(w)"]
		65 -> 64 [label=""]
		64 -> 65 [label="range(l)"]
		66 [label="result[i][j][k] = sum_val / (l * w)
"]
		66 -> 60 [label=""]
		64 -> 66 [label=""]
		62 -> 64 [label=""]
		60 -> 62 [label="range(res_w)"]
		60 -> 58 [label=""]
		58 -> 60 [label="range(res_l)"]
		58 -> 57 [label=""]
		57 -> 58 [label="range(len(input))"]
		59 [label="return result
"]
		57 -> 59 [label=""]
		56 -> 57 [label=""]
		55 -> 56 [label=""]
		54 -> 55 [label="zero_pad > 0"]
		54 -> 56 [label="(zero_pad <= 0)"]
	}
	subgraph clusterreLU {
		graph [label=reLU]
		72 [label="for i in range(len(img)):
"]
		73 [label="for j in range(len(img[0])):
"]
		75 [label="for k in range(len(img[0][0])):
"]
		77 [label="img[i][j][k] = max(img[i][j][k], 0)
"]
		"77_calls" [label=max shape=box]
		77 -> "77_calls" [label=calls style=dashed]
		77 -> 75 [label=""]
		75 -> 77 [label="range(len(img[0][0]))"]
		75 -> 73 [label=""]
		73 -> 75 [label="range(len(img[0]))"]
		73 -> 72 [label=""]
		72 -> 73 [label="range(len(img))"]
		74 [label="return img
"]
		72 -> 74 [label=""]
	}
	subgraph clusterget_mean {
		graph [label=get_mean]
		82 [label="sum_val = 0
"]
		83 [label="for i in range(len(row)):
"]
		84 [label="sum_val += row[i]
"]
		84 -> 83 [label=""]
		83 -> 84 [label="range(len(row))"]
		85 [label="return sum_val / len(row)
"]
		83 -> 85 [label=""]
		82 -> 83 [label=""]
	}
	subgraph clusterstd_dev {
		graph [label=std_dev]
		89 [label="result = 0
"]
		90 [label="for i in range(len(row)):
"]
		91 [label="diff = row[i] - get_mean(row)
result += diff * diff
"]
		"91_calls" [label=get_mean shape=box]
		91 -> "91_calls" [label=calls style=dashed]
		91 -> 90 [label=""]
		90 -> 91 [label="range(len(row))"]
		92 [label="return math.sqrt(result / len(row))
"]
		90 -> 92 [label=""]
		89 -> 90 [label=""]
	}
	subgraph clusterBN_layer {
		graph [label=BN_layer]
		96 [label="for i in range(channels):
"]
		97 [label="for j in range(len(img[0])):
"]
		99 [label="dev = std_dev(img[i][j][:])
mean = get_mean(img[i][j][:])
if dev == 0.0:
"]
		"99_calls" [label="std_dev
get_mean" shape=box]
		99 -> "99_calls" [label=calls style=dashed]
		101 [label="dev = 1.0
"]
		102 [label="for k in range(len(img[0][0])):
"]
		103 [label="img[i][j][k] = weights[j] * ((img[i][j][k] - mean) / dev) + biases[j]
"]
		103 -> 102 [label=""]
		102 -> 103 [label="range(len(img[0][0]))"]
		102 -> 97 [label=""]
		101 -> 102 [label=""]
		99 -> 101 [label="dev == 0.0"]
		99 -> 102 [label="(dev != 0.0)"]
		97 -> 99 [label="range(len(img[0]))"]
		97 -> 96 [label=""]
		96 -> 97 [label="range(channels)"]
		98 [label="return img
"]
		96 -> 98 [label=""]
	}
	subgraph clusterflatten_layer {
		graph [label=flatten_layer]
		108 [label="result = np.zeros(len(img) * len(img[0]) * len(img[0][0]))
index = 0
"]
		"108_calls" [label="np.zeros" shape=box]
		108 -> "108_calls" [label=calls style=dashed]
		109 [label="for i in range(len(img)):
"]
		110 [label="for j in range(len(img[0])):
"]
		112 [label="for k in range(len(img[0][0])):
"]
		114 [label="result[index] = img[i][j][k]
index += 1
"]
		114 -> 112 [label=""]
		112 -> 114 [label="range(len(img[0][0]))"]
		112 -> 110 [label=""]
		110 -> 112 [label="range(len(img[0]))"]
		110 -> 109 [label=""]
		109 -> 110 [label="range(len(img))"]
		111 [label="return result
"]
		109 -> 111 [label=""]
		108 -> 109 [label=""]
	}
	subgraph clusterfc_layer {
		graph [label=fc_layer]
		119 [label="result = np.zeros(len(W[0]))
"]
		"119_calls" [label="np.zeros" shape=box]
		119 -> "119_calls" [label=calls style=dashed]
		120 [label="for i in range(len(W[0])):
"]
		121 [label="sum_val = W_0[i]
"]
		123 [label="for j in range(len(arr)):
"]
		124 [label="sum_val += arr[j] * W[j][i]
"]
		124 -> 123 [label=""]
		123 -> 124 [label="range(len(arr))"]
		125 [label="result[i] = sum_val
"]
		125 -> 120 [label=""]
		123 -> 125 [label=""]
		121 -> 123 [label=""]
		120 -> 121 [label="range(len(W[0]))"]
		122 [label="return result
"]
		120 -> 122 [label=""]
		119 -> 120 [label=""]
	}
	subgraph clustersoftmax {
		graph [label=softmax]
		129 [label="sum_val = 0
"]
		130 [label="for i in range(len(img)):
"]
		131 [label="sum_val += math.exp(img[i])
"]
		"131_calls" [label="math.exp" shape=box]
		131 -> "131_calls" [label=calls style=dashed]
		131 -> 130 [label=""]
		130 -> 131 [label="range(len(img))"]
		132 [label="result = np.zeros(len(img))
"]
		"132_calls" [label="np.zeros" shape=box]
		132 -> "132_calls" [label=calls style=dashed]
		133 [label="for i in range(len(img)):
"]
		134 [label="result[i] = math.exp(img[i]) / sum_val
"]
		"134_calls" [label="math.exp" shape=box]
		134 -> "134_calls" [label=calls style=dashed]
		134 -> 133 [label=""]
		133 -> 134 [label="range(len(img))"]
		135 [label="return result
"]
		133 -> 135 [label=""]
		132 -> 133 [label=""]
		130 -> 132 [label=""]
		129 -> 130 [label=""]
	}
	subgraph clustermain {
		graph [label=main]
		139 [label="zero_pad = 3
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
		"139_calls" [label="np.random.rand
np.random.rand
conv_layer
np.random.rand
np.random.rand
BN_layer
reLU
max_pool
np.random.rand" shape=box]
		139 -> "139_calls" [label=calls style=dashed]
		140 [label="for i in range(2):
"]
		141 [label="byPass = np.copy(img)
img = conv_layer(img, filt, 1, 3, 1)
img = BN_layer(img, 1, weights, biases)
img = reLU(img)
img = conv_layer(img, filt, 1, 3, 1)
img += byPass
img = reLU(img)
"]
		"141_calls" [label="np.copy
conv_layer
BN_layer
reLU
conv_layer
reLU" shape=box]
		141 -> "141_calls" [label=calls style=dashed]
		141 -> 140 [label=""]
		140 -> 141 [label="range(2)"]
		142 [label="filt = np.random.rand(1, len(filt[0]), len(filt[0][0]))
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
print(final)
return 0
"]
		"142_calls" [label="np.random.rand
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
softmax
print" shape=box]
		142 -> "142_calls" [label=calls style=dashed]
		140 -> 142 [label=""]
		139 -> 140 [label=""]
	}
}
