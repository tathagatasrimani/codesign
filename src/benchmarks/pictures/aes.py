digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="Sbox = (99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 
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
InvSbox = (82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 
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
Rcon = (0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154, 47,
    94, 188, 99, 198, 151, 53, 106, 212, 179, 125, 250, 239, 197, 145, 57)
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
	102 [label="import time
start = time.time()
aes = AES(1212304810341341)
aes.encrypt(1212304810341341)
end = time.time()
"]
	"102_calls" [label="time.time
AES
aes.encrypt
time.time" shape=box]
	102 -> "102_calls" [label=calls style=dashed]
	1 -> 102 [label="__name__ == '__main__'"]
	subgraph clusterxtime {
		graph [label=xtime]
		3 [label="return (a << 1 ^ 27) & 255 if a & 128 else a << 1
"]
	}
	subgraph clustertext2matrix {
		graph [label=text2matrix]
		7 [label="matrix = []
"]
		8 [label="for i in range(16):
"]
		9 [label="byte = text >> 8 * (15 - i) & 255
if i % 4 == 0:
"]
		11 [label="matrix.append([byte])
"]
		"11_calls" [label="matrix.append" shape=box]
		11 -> "11_calls" [label=calls style=dashed]
		11 -> 8 [label=""]
		9 -> 11 [label="i % 4 == 0"]
		13 [label="matrix[int(i / 4)].append(byte)
"]
		"13_calls" [label="matrix.append" shape=box]
		13 -> "13_calls" [label=calls style=dashed]
		13 -> 8 [label=""]
		9 -> 13 [label="(i % 4 != 0)"]
		8 -> 9 [label="range(16)"]
		10 [label="return matrix
"]
		8 -> 10 [label=""]
		7 -> 8 [label=""]
	}
	subgraph clustermatrix2text {
		graph [label=matrix2text]
		17 [label="text = 0
"]
		18 [label="for i in range(4):
"]
		19 [label="for j in range(4):
"]
		21 [label="text |= matrix[i][j] << 120 - 8 * (4 * i + j)
"]
		21 -> 19 [label=""]
		19 -> 21 [label="range(4)"]
		19 -> 18 [label=""]
		18 -> 19 [label="range(4)"]
		20 [label="return text
"]
		18 -> 20 [label=""]
		17 -> 18 [label=""]
	}
	subgraph cluster__init__ {
		graph [label=__init__]
		26 [label="self.change_key(master_key)
"]
		"26_calls" [label="self.change_key" shape=box]
		26 -> "26_calls" [label=calls style=dashed]
	}
	subgraph clusterchange_key {
		graph [label=change_key]
		29 [label="self.round_keys = text2matrix(master_key)
"]
		"29_calls" [label=text2matrix shape=box]
		29 -> "29_calls" [label=calls style=dashed]
		30 [label="for i in range(4, 4 * 11):
"]
		31 [label="self.round_keys.append([])
if i % 4 == 0:
"]
		"31_calls" [label="self.round_keys.append" shape=box]
		31 -> "31_calls" [label=calls style=dashed]
		33 [label="byte = self.round_keys[i - 4][0] ^ Sbox[self.round_keys[i - 1][1]] ^ Rcon[int
    (i / 4)]
self.round_keys[i].append(byte)
"]
		"33_calls" [label="int
round_keys.append" shape=box]
		33 -> "33_calls" [label=calls style=dashed]
		38 [label="for j in range(1, 4):
"]
		39 [label="byte = self.round_keys[i - 4][j] ^ Sbox[self.round_keys[i - 1][(j + 1) % 4]]
self.round_keys[i].append(byte)
"]
		"39_calls" [label="round_keys.append" shape=box]
		39 -> "39_calls" [label=calls style=dashed]
		39 -> 38 [label=""]
		38 -> 39 [label="range(1, 4)"]
		38 -> 30 [label=""]
		33 -> 38 [label=""]
		31 -> 33 [label="i % 4 == 0"]
		35 [label="for j in range(4):
"]
		36 [label="byte = self.round_keys[i - 4][j] ^ self.round_keys[i - 1][j]
self.round_keys[i].append(byte)
"]
		"36_calls" [label="round_keys.append" shape=box]
		36 -> "36_calls" [label=calls style=dashed]
		36 -> 35 [label=""]
		35 -> 36 [label="range(4)"]
		35 -> 30 [label=""]
		31 -> 35 [label="(i % 4 != 0)"]
		30 -> 31 [label="range(4, 4 * 11)"]
		29 -> 30 [label=""]
	}
	subgraph clusterencrypt {
		graph [label=encrypt]
		43 [label="self.plain_state = text2matrix(plaintext)
self.__add_round_key(self.plain_state, self.round_keys[:4])
"]
		"43_calls" [label="text2matrix
self.__add_round_key" shape=box]
		43 -> "43_calls" [label=calls style=dashed]
		44 [label="for i in range(1, 10):
"]
		45 [label="self.__round_encrypt(self.plain_state, self.round_keys[4 * i:4 * (i + 1)])
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
		50 [label="self.cipher_state = text2matrix(ciphertext)
self.__add_round_key(self.cipher_state, self.round_keys[40:])
self.__inv_shift_rows(self.cipher_state)
self.__inv_sub_bytes(self.cipher_state)
"]
		"50_calls" [label="text2matrix
self.__add_round_key
self.__inv_shift_rows
self.__inv_sub_bytes" shape=box]
		50 -> "50_calls" [label=calls style=dashed]
		51 [label="for i in range(9, 0, -1):
"]
		52 [label="self.__round_decrypt(self.cipher_state, self.round_keys[4 * i:4 * (i + 1)])
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
		57 [label="for i in range(4):
"]
		58 [label="for j in range(4):
"]
		60 [label="s[i][j] ^= k[i][j]
"]
		60 -> 58 [label=""]
		58 -> 60 [label="range(4)"]
		58 -> 57 [label=""]
		57 -> 58 [label="range(4)"]
	}
	subgraph cluster__round_encrypt {
		graph [label=__round_encrypt]
		64 [label="self.__sub_bytes(state_matrix)
self.__shift_rows(state_matrix)
self.__mix_columns(state_matrix)
self.__add_round_key(state_matrix, key_matrix)
"]
		"64_calls" [label="self.__sub_bytes
self.__shift_rows
self.__mix_columns
self.__add_round_key" shape=box]
		64 -> "64_calls" [label=calls style=dashed]
	}
	subgraph cluster__round_decrypt {
		graph [label=__round_decrypt]
		67 [label="self.__add_round_key(state_matrix, key_matrix)
self.__inv_mix_columns(state_matrix)
self.__inv_shift_rows(state_matrix)
self.__inv_sub_bytes(state_matrix)
"]
		"67_calls" [label="self.__add_round_key
self.__inv_mix_columns
self.__inv_shift_rows
self.__inv_sub_bytes" shape=box]
		67 -> "67_calls" [label=calls style=dashed]
	}
	subgraph cluster__sub_bytes {
		graph [label=__sub_bytes]
		70 [label="for i in range(4):
"]
		71 [label="for j in range(4):
"]
		73 [label="s[i][j] = Sbox[s[i][j]]
"]
		73 -> 71 [label=""]
		71 -> 73 [label="range(4)"]
		71 -> 70 [label=""]
		70 -> 71 [label="range(4)"]
	}
	subgraph cluster__inv_sub_bytes {
		graph [label=__inv_sub_bytes]
		77 [label="for i in range(4):
"]
		78 [label="for j in range(4):
"]
		80 [label="s[i][j] = InvSbox[s[i][j]]
"]
		80 -> 78 [label=""]
		78 -> 80 [label="range(4)"]
		78 -> 77 [label=""]
		77 -> 78 [label="range(4)"]
	}
	subgraph cluster__shift_rows {
		graph [label=__shift_rows]
		84 [label="s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]
"]
	}
	subgraph cluster__inv_shift_rows {
		graph [label=__inv_shift_rows]
		87 [label="s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]
"]
	}
	subgraph cluster__mix_single_column {
		graph [label=__mix_single_column]
		90 [label="t = a[0] ^ a[1] ^ a[2] ^ a[3]
u = a[0]
a[0] ^= t ^ xtime(a[0] ^ a[1])
a[1] ^= t ^ xtime(a[1] ^ a[2])
a[2] ^= t ^ xtime(a[2] ^ a[3])
a[3] ^= t ^ xtime(a[3] ^ u)
"]
		"90_calls" [label="xtime
xtime
xtime
xtime" shape=box]
		90 -> "90_calls" [label=calls style=dashed]
	}
	subgraph cluster__mix_columns {
		graph [label=__mix_columns]
		93 [label="for i in range(4):
"]
		94 [label="self.__mix_single_column(s[i])
"]
		"94_calls" [label="self.__mix_single_column" shape=box]
		94 -> "94_calls" [label=calls style=dashed]
		94 -> 93 [label=""]
		93 -> 94 [label="range(4)"]
	}
	subgraph cluster__inv_mix_columns {
		graph [label=__inv_mix_columns]
		98 [label="for i in range(4):
"]
		99 [label="u = xtime(xtime(s[i][0] ^ s[i][2]))
v = xtime(xtime(s[i][1] ^ s[i][3]))
s[i][0] ^= u
s[i][1] ^= v
s[i][2] ^= u
s[i][3] ^= v
"]
		"99_calls" [label="xtime
xtime" shape=box]
		99 -> "99_calls" [label=calls style=dashed]
		99 -> 98 [label=""]
		98 -> 99 [label="range(4)"]
		100 [label="self.__mix_columns(s)
"]
		"100_calls" [label="self.__mix_columns" shape=box]
		100 -> "100_calls" [label=calls style=dashed]
		98 -> 100 [label=""]
	}
}
