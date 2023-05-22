import sys
from instrument_lib import *
print(1, 1)
Sbox = (99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 
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
print(1, 260)
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


def xtime(a):
    print(1, 521)
    return (a << 1 ^ 27) & 255 if a & 128 else a << 1


print(1, 525)
Rcon = (0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154, 47,
    94, 188, 99, 198, 151, 53, 106, 212, 179, 125, 250, 239, 197, 145, 57)


def text2matrix(text):
    print(1, 561)
    print(7, 562)
    matrix = []
    print(8, 563)
    for i in range(16):
        print(8, 563)
        print(9, 564)
        byte = text >> 8 * (15 - i) & 255
        if i % 4 == 0:
            print(9, 565)
            matrix.append([byte])
        else:
            print(9, 565)
            matrix[int(i / 4)].append(byte)
    return matrix


def matrix2text(matrix):
    print(1, 572)
    print(17, 573)
    text = 0
    print(18, 574)
    for i in range(4):
        print(18, 574)
        print(19, 575)
        for j in range(4):
            print(19, 575)
            print(21, 576)
            text |= matrix[i][j] << 120 - 8 * (4 * i + j)
    return text


class AES:

    def __init__(self, master_key):
        print(1, 581)
        self.change_key(master_key)

    def change_key(self, master_key):
        print(1, 584)
        print(29, 585)
        self.round_keys = text2matrix(master_key)
        print(30, 588)
        for i in range(4, 4 * 11):
            print(30, 588)
            self.round_keys.append([])
            if i % 4 == 0:
                print(31, 590)
                print(33, 591)
                byte = self.round_keys[i - 4][0] ^ Sbox[self.round_keys[i -
                    1][1]] ^ Rcon[int(i / 4)]
                self.round_keys[i].append(byte)
                print(38, 598)
                for j in range(1, 4):
                    print(38, 598)
                    print(39, 599)
                    byte = self.round_keys[i - 4][j] ^ Sbox[self.round_keys
                        [i - 1][(j + 1) % 4]]
                    self.round_keys[i].append(byte)
            else:
                print(31, 590)
                print(35, 605)
                for j in range(4):
                    print(35, 605)
                    print(36, 606)
                    byte = self.round_keys[i - 4][j] ^ self.round_keys[i - 1][j
                        ]
                    self.round_keys[i].append(byte)

    def encrypt(self, plaintext):
        print(1, 611)
        print(43, 612)
        self.plain_state = text2matrix(plaintext)
        self.__add_round_key(self.plain_state, self.round_keys[:4])
        print(44, 616)
        for i in range(1, 10):
            print(44, 616)
            self.__round_encrypt(self.plain_state, self.round_keys[4 * i:4 *
                (i + 1)])
        self.__sub_bytes(self.plain_state)
        self.__shift_rows(self.plain_state)
        self.__add_round_key(self.plain_state, self.round_keys[40:])
        return matrix2text(self.plain_state)

    def decrypt(self, ciphertext):
        print(1, 625)
        print(50, 626)
        self.cipher_state = text2matrix(ciphertext)
        self.__add_round_key(self.cipher_state, self.round_keys[40:])
        self.__inv_shift_rows(self.cipher_state)
        self.__inv_sub_bytes(self.cipher_state)
        print(51, 632)
        for i in range(9, 0, -1):
            print(51, 632)
            self.__round_decrypt(self.cipher_state, self.round_keys[4 * i:4 *
                (i + 1)])
        self.__add_round_key(self.cipher_state, self.round_keys[:4])
        return matrix2text(self.cipher_state)

    def __add_round_key(self, s, k):
        print(1, 641)
        print(57, 642)
        for i in range(4):
            print(57, 642)
            print(58, 643)
            for j in range(4):
                print(58, 643)
                print(60, 644)
                s[i][j] ^= k[i][j]

    def __round_encrypt(self, state_matrix, key_matrix):
        print(1, 646)
        self.__sub_bytes(state_matrix)
        self.__shift_rows(state_matrix)
        self.__mix_columns(state_matrix)
        self.__add_round_key(state_matrix, key_matrix)

    def __round_decrypt(self, state_matrix, key_matrix):
        print(1, 652)
        self.__add_round_key(state_matrix, key_matrix)
        self.__inv_mix_columns(state_matrix)
        self.__inv_shift_rows(state_matrix)
        self.__inv_sub_bytes(state_matrix)

    def __sub_bytes(self, s):
        print(1, 658)
        print(70, 659)
        for i in range(4):
            print(70, 659)
            print(71, 660)
            for j in range(4):
                print(71, 660)
                print(73, 661)
                s[i][j] = Sbox[s[i][j]]

    def __inv_sub_bytes(self, s):
        print(1, 663)
        print(77, 664)
        for i in range(4):
            print(77, 664)
            print(78, 665)
            for j in range(4):
                print(78, 665)
                print(80, 666)
                s[i][j] = InvSbox[s[i][j]]

    def __shift_rows(self, s):
        print(1, 668)
        print(84, 669)
        s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
        print(84, 670)
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        print(84, 671)
        s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]

    def __inv_shift_rows(self, s):
        print(1, 673)
        print(87, 674)
        s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
        print(87, 675)
        s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
        print(87, 676)
        s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]

    def __mix_single_column(self, a):
        print(1, 678)
        print(90, 680)
        t = a[0] ^ a[1] ^ a[2] ^ a[3]
        print(90, 681)
        u = a[0]
        print(90, 682)
        a[0] ^= t ^ xtime(a[0] ^ a[1])
        print(90, 683)
        a[1] ^= t ^ xtime(a[1] ^ a[2])
        print(90, 684)
        a[2] ^= t ^ xtime(a[2] ^ a[3])
        print(90, 685)
        a[3] ^= t ^ xtime(a[3] ^ u)

    def __mix_columns(self, s):
        print(1, 687)
        print(93, 688)
        for i in range(4):
            print(93, 688)
            self.__mix_single_column(s[i])

    def __inv_mix_columns(self, s):
        print(1, 691)
        print(98, 693)
        for i in range(4):
            print(98, 693)
            print(99, 694)
            u = xtime(xtime(s[i][0] ^ s[i][2]))
            print(99, 695)
            v = xtime(xtime(s[i][1] ^ s[i][3]))
            print(99, 696)
            s[i][0] ^= u
            print(99, 697)
            s[i][1] ^= v
            print(99, 698)
            s[i][2] ^= u
            print(99, 699)
            s[i][3] ^= v
        self.__mix_columns(s)


if __name__ == '__main__':
    print(1, 703)
    import time
    print(102, 705)
    start = time.time()
    print(102, 706)
    aes = AES(1212304810341341)
    aes.encrypt(1212304810341341)
    print(102, 708)
    end = time.time()
    print(end - start)
else:
    print(1, 703)
