digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="from loop import loop
import numpy as np
Sbox_0 = (99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 
    171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156,
    164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 
    113, 216, 49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 
    235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 
    41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57,
    74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 
    80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 
    33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126,
    61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184,
    20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98,
    145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244,
    234, 101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 
    116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 
    87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155,
    30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230, 66, 104, 
    65, 153, 45, 15, 176, 84, 187, 22)
InvSbox_0 = (82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243,
    215, 251, 124, 227, 57, 130, 155, 47, 255, 135, 52, 142, 67, 68, 196, 
    222, 233, 203, 84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66,
    250, 195, 78, 8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109,
    139, 209, 37, 114, 248, 246, 100, 134, 104, 152, 22, 212, 164, 92, 204,
    93, 101, 182, 146, 108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87,
    167, 141, 157, 132, 144, 216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 
    5, 184, 179, 69, 6, 208, 44, 30, 143, 202, 63, 15, 2, 193, 175, 189, 3,
    1, 19, 138, 107, 58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206,
    240, 180, 230, 115, 150, 172, 116, 34, 231, 173, 53, 133, 226, 249, 55,
    232, 28, 117, 223, 110, 71, 241, 26, 113, 29, 41, 197, 137, 111, 183, 
    98, 14, 170, 24, 190, 27, 252, 86, 62, 75, 198, 210, 121, 32, 154, 219,
    192, 254, 120, 205, 90, 244, 31, 221, 168, 51, 136, 7, 199, 49, 177, 18,
    16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 
    122, 159, 147, 201, 156, 239, 160, 224, 59, 77, 174, 42, 245, 176, 200,
    235, 187, 60, 131, 83, 153, 97, 23, 43, 4, 126, 186, 119, 214, 38, 225,
    105, 20, 99, 85, 33, 12, 125)
def xtime(a):...
Rcon_0 = (0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154, 
    47, 94, 188, 99, 198, 151, 53, 106, 212, 179, 125, 250, 239, 197, 145, 57)
def text2matrix(text):...
def matrix2text(matrix):...
def __init__(self, master_key):...
def change_key(self, master_key):...
def encrypt(self, plaintext):...
def decrypt(self, ciphertext):...
def __add_round_key(self, s, k):...
def __round_encrypt(self, state_matrix, key_matrix):...
def __round_decrypt(self, state_matrix, key_matrix):...
def __sub_bytes(self, s):...
def __inv_sub_bytes(self, s):...
def __shift_rows(self, s):...
def __inv_shift_rows(self, s):...
def __mix_single_column(self, a):...
def __mix_columns(self, s):...
def __inv_mix_columns(self, s):...
if __name__ == '__main__':
"]
	107 [label="import time
start_0 = time.time()
"]
	"107_calls" [label="time.time" shape=box]
	107 -> "107_calls" [label=calls style=dashed]
	109 [label="for i_0 in range(10):
"]
	110 [label="Sbox_0 += Sbox_0
"]
	110 -> 109 [label=""]
	109 -> 110 [label="range(10)"]
	111 [label="Sbox_new_0 = Sbox_0
"]
	112 [label="for i_0 in range(1):
"]
	113 [label="aes_0 = AES(1212304810341341)
aes_0.encrypt(1212304810341341)
"]
	"113_calls" [label="AES
aes_0.encrypt" shape=box]
	113 -> "113_calls" [label=calls style=dashed]
	113 -> 112 [label=""]
	112 -> 113 [label="range(1)"]
	114 [label="end_0 = time.time()
"]
	"114_calls" [label="time.time" shape=box]
	114 -> "114_calls" [label=calls style=dashed]
	112 -> 114 [label=""]
	111 -> 112 [label=""]
	109 -> 111 [label=""]
	107 -> 109 [label=""]
	1 -> 107 [label="__name__ == '__main__'"]
	subgraph clusterxtime {
		graph [label=xtime]
		3 [label="a_1 = a
return (a_1 << 1 ^ 27) & 255 if a_1 & 128 else a_1 << 1
"]
	}
	subgraph clustertext2matrix {
		graph [label=text2matrix]
		7 [label="text_2 = text
matrix_2 = []
"]
		8 [label="for i_2 in range(16):
"]
		9 [label="byte_2 = text_2 >> 8 * (15 - i_2) & 255
if i_2 % 4 == 0:
"]
		11 [label="matrix_2.append([byte_2])
"]
		"11_calls" [label="matrix_2.append" shape=box]
		11 -> "11_calls" [label=calls style=dashed]
		11 -> 8 [label=""]
		9 -> 11 [label="i_2 % 4 == 0"]
		13 [label="matrix_2[int(i_2 / 4)].append(byte_2)
"]
		"13_calls" [label="matrix_2.append" shape=box]
		13 -> "13_calls" [label=calls style=dashed]
		13 -> 8 [label=""]
		9 -> 13 [label="(i_2 % 4 != 0)"]
		8 -> 9 [label="range(16)"]
		10 [label="return matrix_2
"]
		8 -> 10 [label=""]
		7 -> 8 [label=""]
	}
	subgraph clustermatrix2text {
		graph [label=matrix2text]
		17 [label="matrix_3 = matrix
text_3 = 0
"]
		18 [label="for i_3 in range(4):
"]
		19 [label="for j_3 in range(4):
"]
		21 [label="text_3 |= matrix_3[i_3][j_3] << 120 - 8 * (4 * i_3 + j_3)
"]
		21 -> 19 [label=""]
		19 -> 21 [label="range(4)"]
		19 -> 18 [label=""]
		18 -> 19 [label="range(4)"]
		20 [label="return text_3
"]
		18 -> 20 [label=""]
		17 -> 18 [label=""]
	}
	subgraph cluster__init__ {
		graph [label=__init__]
		26 [label="self = self
master_key_4 = master_key
self.change_key(master_key_4)
"]
		"26_calls" [label="self.change_key" shape=box]
		26 -> "26_calls" [label=calls style=dashed]
	}
	subgraph clusterchange_key {
		graph [label=change_key]
		29 [label="self = self
master_key_5 = master_key
self.round_keys = text2matrix(master_key_5)
"]
		"29_calls" [label=text2matrix shape=box]
		29 -> "29_calls" [label=calls style=dashed]
		30 [label="for i_5 in range(4, 4 * 11):
"]
		31 [label="self.round_keys.append([])
if i_5 % 4 == 0:
"]
		"31_calls" [label="self.round_keys.append" shape=box]
		31 -> "31_calls" [label=calls style=dashed]
		33 [label="byte_5 = self.round_keys[i_5 - 4][0] ^ Sbox_0[self.round_keys[i_5 - 1][1]
    ] ^ Rcon_0[int(i_5 / 4)]
self.round_keys[i_5].append(byte_5)
"]
		"33_calls" [label="int
round_keys.append" shape=box]
		33 -> "33_calls" [label=calls style=dashed]
		38 [label="for j_5 in range(1, 4):
"]
		39 [label="byte_5 = self.round_keys[i_5 - 4][j_5] ^ Sbox_0[self.round_keys[i_5 - 1][(
    j_5 + 1) % 4]]
self.round_keys[i_5].append(byte_5)
"]
		"39_calls" [label="round_keys.append" shape=box]
		39 -> "39_calls" [label=calls style=dashed]
		39 -> 38 [label=""]
		38 -> 39 [label="range(1, 4)"]
		38 -> 30 [label=""]
		33 -> 38 [label=""]
		31 -> 33 [label="i_5 % 4 == 0"]
		35 [label="for j_5 in range(4):
"]
		36 [label="byte_5 = self.round_keys[i_5 - 4][j_5] ^ self.round_keys[i_5 - 1][j_5]
self.round_keys[i_5].append(byte_5)
"]
		"36_calls" [label="round_keys.append" shape=box]
		36 -> "36_calls" [label=calls style=dashed]
		36 -> 35 [label=""]
		35 -> 36 [label="range(4)"]
		35 -> 30 [label=""]
		31 -> 35 [label="(i_5 % 4 != 0)"]
		30 -> 31 [label="range(4, 4 * 11)"]
		29 -> 30 [label=""]
	}
	subgraph clusterencrypt {
		graph [label=encrypt]
		43 [label="self = self
plaintext_6 = plaintext
self.plain_state = text2matrix(plaintext_6)
self.__add_round_key(self.plain_state, self.round_keys[:4])
"]
		"43_calls" [label="text2matrix
self.__add_round_key" shape=box]
		43 -> "43_calls" [label=calls style=dashed]
		44 [label="for i_6 in range(1, 10):
"]
		45 [label="self.__round_encrypt(self.plain_state, self.round_keys[4 * i_6:4 * (i_6 + 1)])
"]
		"45_calls" [label="self.__round_encrypt" shape=box]
		45 -> "45_calls" [label=calls style=dashed]
		45 -> 44 [label=""]
		44 -> 45 [label="range(1, 10)"]
		46 [label="self.__sub_bytes(self.plain_state)
self.__shift_rows(self.plain_state)
self.__add_round_key(self.plain_state, self.round_keys[40:])
return matrix2text(self.plain_state)
"]
		"46_calls" [label="self.__sub_bytes
self.__shift_rows
self.__add_round_key" shape=box]
		46 -> "46_calls" [label=calls style=dashed]
		44 -> 46 [label=""]
		43 -> 44 [label=""]
	}
	subgraph clusterdecrypt {
		graph [label=decrypt]
		50 [label="self = self
ciphertext_7 = ciphertext
self.cipher_state = text2matrix(ciphertext_7)
self.__add_round_key(self.cipher_state, self.round_keys[40:])
self.__inv_shift_rows(self.cipher_state)
self.__inv_sub_bytes(self.cipher_state)
"]
		"50_calls" [label="text2matrix
self.__add_round_key
self.__inv_shift_rows
self.__inv_sub_bytes" shape=box]
		50 -> "50_calls" [label=calls style=dashed]
		51 [label="for i_7 in range(9, 0, -1):
"]
		52 [label="self.__round_decrypt(self.cipher_state, self.round_keys[4 * i_7:4 * (i_7 + 1)])
"]
		"52_calls" [label="self.__round_decrypt" shape=box]
		52 -> "52_calls" [label=calls style=dashed]
		52 -> 51 [label=""]
		51 -> 52 [label="range(9, 0, -1)"]
		53 [label="self.__add_round_key(self.cipher_state, self.round_keys[:4])
return matrix2text(self.cipher_state)
"]
		"53_calls" [label="self.__add_round_key" shape=box]
		53 -> "53_calls" [label=calls style=dashed]
		51 -> 53 [label=""]
		50 -> 51 [label=""]
	}
	subgraph cluster__add_round_key {
		graph [label=__add_round_key]
		57 [label="self = self
s_8 = s
k_8 = k
"]
		58 [label="for i_8 in range(4):
"]
		59 [label="for j_8 in range(4):
"]
		61 [label="s_8[i_8][j_8] ^= k_8[i_8][j_8]
"]
		61 -> 59 [label=""]
		59 -> 61 [label="range(4)"]
		59 -> 58 [label=""]
		58 -> 59 [label="range(4)"]
		57 -> 58 [label=""]
	}
	subgraph cluster__round_encrypt {
		graph [label=__round_encrypt]
		65 [label="self = self
state_matrix_9 = state_matrix
key_matrix_9 = key_matrix
self.__sub_bytes(state_matrix_9)
self.__shift_rows(state_matrix_9)
self.__mix_columns(state_matrix_9)
self.__add_round_key(state_matrix_9, key_matrix_9)
"]
		"65_calls" [label="self.__sub_bytes
self.__shift_rows
self.__mix_columns
self.__add_round_key" shape=box]
		65 -> "65_calls" [label=calls style=dashed]
	}
	subgraph cluster__round_decrypt {
		graph [label=__round_decrypt]
		68 [label="self = self
state_matrix_10 = state_matrix
key_matrix_10 = key_matrix
self.__add_round_key(state_matrix_10, key_matrix_10)
self.__inv_mix_columns(state_matrix_10)
self.__inv_shift_rows(state_matrix_10)
self.__inv_sub_bytes(state_matrix_10)
"]
		"68_calls" [label="self.__add_round_key
self.__inv_mix_columns
self.__inv_shift_rows
self.__inv_sub_bytes" shape=box]
		68 -> "68_calls" [label=calls style=dashed]
	}
	subgraph cluster__sub_bytes {
		graph [label=__sub_bytes]
		71 [label="self = self
s_11 = s
"]
		72 [label="for i_11 in range(4):
"]
		73 [label="for j_11 in range(4):
"]
		75 [label="s_11[i_11][j_11] = Sbox_0[s_11[i_11][j_11]]
"]
		75 -> 73 [label=""]
		73 -> 75 [label="range(4)"]
		73 -> 72 [label=""]
		72 -> 73 [label="range(4)"]
		71 -> 72 [label=""]
	}
	subgraph cluster__inv_sub_bytes {
		graph [label=__inv_sub_bytes]
		79 [label="self = self
s_12 = s
"]
		80 [label="for i_12 in range(4):
"]
		81 [label="for j_12 in range(4):
"]
		83 [label="s_12[i_12][j_12] = InvSbox_0[s_12[i_12][j_12]]
"]
		83 -> 81 [label=""]
		81 -> 83 [label="range(4)"]
		81 -> 80 [label=""]
		80 -> 81 [label="range(4)"]
		79 -> 80 [label=""]
	}
	subgraph cluster__shift_rows {
		graph [label=__shift_rows]
		87 [label="self = self
s_13 = s
s_13[0][1], s_13[1][1], s_13[2][1], s_13[3][1] = s_13[1][1], s_13[2][1], s_13[3
    ][1], s_13[0][1]
s_13[0][2], s_13[1][2], s_13[2][2], s_13[3][2] = s_13[2][2], s_13[3][2], s_13[0
    ][2], s_13[1][2]
s_13[0][3], s_13[1][3], s_13[2][3], s_13[3][3] = s_13[3][3], s_13[0][3], s_13[1
    ][3], s_13[2][3]
"]
	}
	subgraph cluster__inv_shift_rows {
		graph [label=__inv_shift_rows]
		90 [label="self = self
s_14 = s
s_14[0][1], s_14[1][1], s_14[2][1], s_14[3][1] = s_14[3][1], s_14[0][1], s_14[1
    ][1], s_14[2][1]
s_14[0][2], s_14[1][2], s_14[2][2], s_14[3][2] = s_14[2][2], s_14[3][2], s_14[0
    ][2], s_14[1][2]
s_14[0][3], s_14[1][3], s_14[2][3], s_14[3][3] = s_14[1][3], s_14[2][3], s_14[3
    ][3], s_14[0][3]
"]
	}
	subgraph cluster__mix_single_column {
		graph [label=__mix_single_column]
		93 [label="self = self
a_15 = a
t_15 = a_15[0] ^ a_15[1] ^ a_15[2] ^ a_15[3]
u_15 = a_15[0]
a_15[0] ^= t_15 ^ xtime(a_15[0] ^ a_15[1])
a_15[1] ^= t_15 ^ xtime(a_15[1] ^ a_15[2])
a_15[2] ^= t_15 ^ xtime(a_15[2] ^ a_15[3])
a_15[3] ^= t_15 ^ xtime(a_15[3] ^ u_15)
"]
		"93_calls" [label="xtime
xtime
xtime
xtime" shape=box]
		93 -> "93_calls" [label=calls style=dashed]
	}
	subgraph cluster__mix_columns {
		graph [label=__mix_columns]
		96 [label="self = self
s_16 = s
"]
		97 [label="for i_16 in range(4):
"]
		98 [label="self.__mix_single_column(s_16[i_16])
"]
		"98_calls" [label="self.__mix_single_column" shape=box]
		98 -> "98_calls" [label=calls style=dashed]
		98 -> 97 [label=""]
		97 -> 98 [label="range(4)"]
		96 -> 97 [label=""]
	}
	subgraph cluster__inv_mix_columns {
		graph [label=__inv_mix_columns]
		102 [label="self = self
s_17 = s
"]
		103 [label="for i_17 in range(4):
"]
		104 [label="u_17 = xtime(xtime(s_17[i_17][0] ^ s_17[i_17][2]))
v_17 = xtime(xtime(s_17[i_17][1] ^ s_17[i_17][3]))
s_17[i_17][0] ^= u_17
s_17[i_17][1] ^= v_17
s_17[i_17][2] ^= u_17
s_17[i_17][3] ^= v_17
"]
		"104_calls" [label="xtime
xtime" shape=box]
		104 -> "104_calls" [label=calls style=dashed]
		104 -> 103 [label=""]
		103 -> 104 [label="range(4)"]
		105 [label="self.__mix_columns(s_17)
"]
		"105_calls" [label="self.__mix_columns" shape=box]
		105 -> "105_calls" [label=calls style=dashed]
		103 -> 105 [label=""]
		102 -> 103 [label=""]
	}
}
