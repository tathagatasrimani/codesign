import sys
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
    print(1, 520)
    if a & 128:
        print(3, 521)
        return (a << 1 ^ 27) & 255
    else:
        print(3, 521)
        return a << 1


print(1, 526)
Rcon = (0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154, 47,
    94, 188, 99, 198, 151, 53, 106, 212, 179, 125, 250, 239, 197, 145, 57)
print(1, 561)
round_keys = []


def text2matrix(text):
    print(1, 563)
    print(11, 564)
    matrix = []
    print(12, 565)
    for i in range(16):
        print(12, 565)
        print(13, 566)
        byte = text >> 8 * (15 - i) & 255
        if i % 4 == 0:
            print(13, 567)
            matrix.append([byte])
        else:
            print(13, 567)
            matrix[int(i / 4)].append(byte)
    return matrix


def matrix2text(matrix):
    print(1, 573)
    print(21, 574)
    text = 0
    print(22, 575)
    for i in range(4):
        print(22, 575)
        print(23, 576)
        for j in range(4):
            print(23, 576)
            print(25, 577)
            text |= matrix[i][j] << 120 - 8 * (4 * i + j)
    return text


def __init__(master_key):
    print(1, 580)
    change_key(master_key)


def change_key(master_key):
    print(1, 583)
    global round_keys
    print(33, 585)
    round_keys = text2matrix(master_key)
    print(34, 588)
    for i in range(4, 4 * 11):
        print(34, 588)
        round_keys.append([])
        if i % 4 == 0:
            print(35, 590)
            print(37, 591)
            byte = round_keys[i - 4][0] ^ Sbox[round_keys[i - 1][1]] ^ Rcon[int
                (i / 4)]
            round_keys[i].append(byte)
            print(42, 594)
            for j in range(1, 4):
                print(42, 594)
                print(43, 595)
                byte = round_keys[i - 4][j] ^ Sbox[round_keys[i - 1][(j + 1
                    ) % 4]]
                round_keys[i].append(byte)
        else:
            print(35, 590)
            print(39, 598)
            for j in range(4):
                print(39, 598)
                print(40, 599)
                byte = round_keys[i - 4][j] ^ round_keys[i - 1][j]
                round_keys[i].append(byte)


def encrypt(plaintext):
    print(1, 603)
    global round_keys
    print(47, 605)
    plain_state = text2matrix(plaintext)
    __add_round_key(plain_state, round_keys[:4])
    print(48, 609)
    for i in range(1, 10):
        print(48, 609)
        __round_encrypt(plain_state, round_keys[4 * i:4 * (i + 1)])
    __sub_bytes(plain_state)
    __shift_rows(plain_state)
    __add_round_key(plain_state, round_keys[40:])
    return matrix2text(plain_state)


def decrypt(ciphertext):
    print(1, 618)
    global round_keys
    print(54, 620)
    cipher_state = text2matrix(ciphertext)
    __add_round_key(cipher_state, round_keys[40:])
    __inv_shift_rows(cipher_state)
    __inv_sub_bytes(cipher_state)
    print(55, 626)
    for i in range(9, 0, -1):
        print(55, 626)
        __round_decrypt(cipher_state, round_keys[4 * i:4 * (i + 1)])
    __add_round_key(cipher_state, round_keys[:4])
    return matrix2text(cipher_state)


def __add_round_key(s, k):
    print(1, 635)
    print(61, 636)
    for i in range(4):
        print(61, 636)
        print(62, 637)
        for j in range(4):
            print(62, 637)
            print(64, 638)
            s[i][j] ^= k[i][j]


def __round_encrypt(state_matrix, key_matrix):
    print(1, 640)
    __sub_bytes(state_matrix)
    __shift_rows(state_matrix)
    __mix_columns(state_matrix)
    __add_round_key(state_matrix, key_matrix)


def __round_decrypt(state_matrix, key_matrix):
    print(1, 646)
    __add_round_key(state_matrix, key_matrix)
    __inv_mix_columns(state_matrix)
    __inv_shift_rows(state_matrix)
    __inv_sub_bytes(state_matrix)


def __sub_bytes(s):
    print(1, 652)
    print(74, 653)
    for i in range(4):
        print(74, 653)
        print(75, 654)
        for j in range(4):
            print(75, 654)
            print(77, 655)
            s[i][j] = Sbox[s[i][j]]


def __inv_sub_bytes(s):
    print(1, 657)
    print(81, 658)
    for i in range(4):
        print(81, 658)
        print(82, 659)
        for j in range(4):
            print(82, 659)
            print(84, 660)
            s[i][j] = InvSbox[s[i][j]]


def __shift_rows(s):
    print(1, 662)
    print(88, 663)
    s[0][1] = s[1][1]
    print(88, 664)
    s[1][1] = s[2][1]
    print(88, 665)
    s[2][1] = s[3][1]
    print(88, 666)
    s[3][1] = s[0][1]
    print(88, 668)
    s[0][2] = s[2][2]
    print(88, 669)
    s[1][2] = s[3][2]
    print(88, 670)
    s[2][2] = s[0][2]
    print(88, 671)
    s[3][2] = s[1][2]
    print(88, 673)
    s[0][3] = s[3][3]
    print(88, 674)
    s[1][3] = s[0][3]
    print(88, 675)
    s[2][3] = s[1][3]
    print(88, 676)
    s[3][3] = s[2][3]


def __inv_shift_rows(s):
    print(1, 678)
    print(91, 679)
    s[0][1] = s[3][1]
    print(91, 680)
    s[1][1] = s[0][1]
    print(91, 681)
    s[2][1] = s[1][1]
    print(91, 682)
    s[3][1] = s[2][1]
    print(91, 684)
    s[0][2] = s[2][2]
    print(91, 685)
    s[1][2] = s[3][2]
    print(91, 686)
    s[2][2] = s[0][2]
    print(91, 687)
    s[3][2] = s[1][2]
    print(91, 689)
    s[0][3] = s[1][3]
    print(91, 690)
    s[1][3] = s[2][3]
    print(91, 691)
    s[2][3] = s[3][3]
    print(91, 692)
    s[3][3] = s[0][3]


def __mix_single_column(a):
    print(1, 694)
    print(94, 696)
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    print(94, 697)
    u = a[0]
    print(94, 698)
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    print(94, 699)
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    print(94, 700)
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    print(94, 701)
    a[3] ^= t ^ xtime(a[3] ^ u)


def __mix_columns(s):
    print(1, 703)
    print(97, 704)
    for i in range(4):
        print(97, 704)
        __mix_single_column(s[i])


def __inv_mix_columns(s):
    print(1, 707)
    print(102, 709)
    for i in range(4):
        print(102, 709)
        print(103, 710)
        u = xtime(xtime(s[i][0] ^ s[i][2]))
        print(103, 711)
        v = xtime(xtime(s[i][1] ^ s[i][3]))
        print(103, 712)
        s[i][0] ^= u
        print(103, 713)
        s[i][1] ^= v
        print(103, 714)
        s[i][2] ^= u
        print(103, 715)
        s[i][3] ^= v
    __mix_columns(s)


if __name__ == '__main__':
    print(1, 719)
    import time
    print(106, 721)
    start = time.time()
    print(106, 722)
    aes = __init__(1212304810341341)
    encrypt(1212304810341341)
    print(106, 724)
    end = time.time()
    print(end - start)
else:
    print(1, 719)
