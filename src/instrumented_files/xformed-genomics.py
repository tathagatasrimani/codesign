import sys
from instrument_lib import *
import numpy as np
import os
print(1, 3)
path_0 = instrument_read(os, 'os').getcwd()
write_instrument_read(path_0, 'path_0')
if type(path_0) == np.ndarray:
    print('malloc', sys.getsizeof(path_0), 'path_0', path_0.shape)
elif type(path_0) == list:
    dims = []
    tmp = path_0
    while type(tmp) == list:
        dims.append(len(tmp))
        if len(tmp) > 0:
            tmp = tmp[0]
        else:
            tmp = None
    print('malloc', sys.getsizeof(path_0), 'path_0', dims)
elif type(path_0) == tuple:
    print('malloc', sys.getsizeof(path_0), 'path_0', [len(path_0)])
else:
    print('malloc', sys.getsizeof(path_0), 'path_0')


def read_fasta(fasta_file):
    print('enter scope 1')
    print(1, 6)
    print(3, 7)
    fasta_file_1 = instrument_read(fasta_file, 'fasta_file')
    write_instrument_read(fasta_file_1, 'fasta_file_1')
    if type(fasta_file_1) == np.ndarray:
        print('malloc', sys.getsizeof(fasta_file_1), 'fasta_file_1',
            fasta_file_1.shape)
    elif type(fasta_file_1) == list:
        dims = []
        tmp = fasta_file_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fasta_file_1), 'fasta_file_1', dims)
    elif type(fasta_file_1) == tuple:
        print('malloc', sys.getsizeof(fasta_file_1), 'fasta_file_1', [len(
            fasta_file_1)])
    else:
        print('malloc', sys.getsizeof(fasta_file_1), 'fasta_file_1')
    """
    Reads a fasta file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(3, 12)
    fasta_dict_1 = {}
    write_instrument_read(fasta_dict_1, 'fasta_dict_1')
    if type(fasta_dict_1) == np.ndarray:
        print('malloc', sys.getsizeof(fasta_dict_1), 'fasta_dict_1',
            fasta_dict_1.shape)
    elif type(fasta_dict_1) == list:
        dims = []
        tmp = fasta_dict_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fasta_dict_1), 'fasta_dict_1', dims)
    elif type(fasta_dict_1) == tuple:
        print('malloc', sys.getsizeof(fasta_dict_1), 'fasta_dict_1', [len(
            fasta_dict_1)])
    else:
        print('malloc', sys.getsizeof(fasta_dict_1), 'fasta_dict_1')
    with open(instrument_read(fasta_file_1, 'fasta_file_1'), 'r') as f_1:
        for line_1 in instrument_read(f_1, 'f_1'):
            print(5, 15)
            line_1 = instrument_read(line_1, 'line_1').strip()
            write_instrument_read(line_1, 'line_1')
            if type(line_1) == np.ndarray:
                print('malloc', sys.getsizeof(line_1), 'line_1', line_1.shape)
            elif type(line_1) == list:
                dims = []
                tmp = line_1
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(line_1), 'line_1', dims)
            elif type(line_1) == tuple:
                print('malloc', sys.getsizeof(line_1), 'line_1', [len(line_1)])
            else:
                print('malloc', sys.getsizeof(line_1), 'line_1')
            if not instrument_read(line_1, 'line_1'):
                continue
            if instrument_read(line_1, 'line_1').startswith('>'):
                print(9, 19)
                active_sequence_name_1 = instrument_read_sub(instrument_read
                    (line_1, 'line_1'), 'line_1', None, 1, None, True)
                write_instrument_read(active_sequence_name_1,
                    'active_sequence_name_1')
                if type(active_sequence_name_1) == np.ndarray:
                    print('malloc', sys.getsizeof(active_sequence_name_1),
                        'active_sequence_name_1', active_sequence_name_1.shape)
                elif type(active_sequence_name_1) == list:
                    dims = []
                    tmp = active_sequence_name_1
                    while type(tmp) == list:
                        dims.append(len(tmp))
                        if len(tmp) > 0:
                            tmp = tmp[0]
                        else:
                            tmp = None
                    print('malloc', sys.getsizeof(active_sequence_name_1),
                        'active_sequence_name_1', dims)
                elif type(active_sequence_name_1) == tuple:
                    print('malloc', sys.getsizeof(active_sequence_name_1),
                        'active_sequence_name_1', [len(active_sequence_name_1)]
                        )
                else:
                    print('malloc', sys.getsizeof(active_sequence_name_1),
                        'active_sequence_name_1')
                if instrument_read(active_sequence_name_1,
                    'active_sequence_name_1') not in instrument_read(
                    fasta_dict_1, 'fasta_dict_1'):
                    print(11, 21)
                    fasta_dict_1[instrument_read(instrument_read(
                        active_sequence_name_1, 'active_sequence_name_1'),
                        'active_sequence_name_1')] = []
                    write_instrument_read_sub(fasta_dict_1, 'fasta_dict_1',
                        instrument_read(instrument_read(
                        active_sequence_name_1, 'active_sequence_name_1'),
                        'active_sequence_name_1'), None, None, False)
                continue
            print(10, 23)
            sequence_1 = instrument_read(line_1, 'line_1')
            write_instrument_read(sequence_1, 'sequence_1')
            if type(sequence_1) == np.ndarray:
                print('malloc', sys.getsizeof(sequence_1), 'sequence_1',
                    sequence_1.shape)
            elif type(sequence_1) == list:
                dims = []
                tmp = sequence_1
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(sequence_1), 'sequence_1', dims)
            elif type(sequence_1) == tuple:
                print('malloc', sys.getsizeof(sequence_1), 'sequence_1', [
                    len(sequence_1)])
            else:
                print('malloc', sys.getsizeof(sequence_1), 'sequence_1')
            instrument_read_sub(instrument_read(fasta_dict_1,
                'fasta_dict_1'), 'fasta_dict_1', instrument_read(
                active_sequence_name_1, 'active_sequence_name_1'), None,
                None, False).append(instrument_read(sequence_1, 'sequence_1'))
    print('exit scope 1')
    return instrument_read(fasta_dict_1, 'fasta_dict_1')
    print('exit scope 1')


def read_fastq(fastq_file):
    print('enter scope 2')
    print(1, 28)
    print(16, 29)
    fastq_file_2 = instrument_read(fastq_file, 'fastq_file')
    write_instrument_read(fastq_file_2, 'fastq_file_2')
    if type(fastq_file_2) == np.ndarray:
        print('malloc', sys.getsizeof(fastq_file_2), 'fastq_file_2',
            fastq_file_2.shape)
    elif type(fastq_file_2) == list:
        dims = []
        tmp = fastq_file_2
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fastq_file_2), 'fastq_file_2', dims)
    elif type(fastq_file_2) == tuple:
        print('malloc', sys.getsizeof(fastq_file_2), 'fastq_file_2', [len(
            fastq_file_2)])
    else:
        print('malloc', sys.getsizeof(fastq_file_2), 'fastq_file_2')
    """
    Reads a fastq file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(16, 34)
    fastq_dict_2 = {}
    write_instrument_read(fastq_dict_2, 'fastq_dict_2')
    if type(fastq_dict_2) == np.ndarray:
        print('malloc', sys.getsizeof(fastq_dict_2), 'fastq_dict_2',
            fastq_dict_2.shape)
    elif type(fastq_dict_2) == list:
        dims = []
        tmp = fastq_dict_2
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fastq_dict_2), 'fastq_dict_2', dims)
    elif type(fastq_dict_2) == tuple:
        print('malloc', sys.getsizeof(fastq_dict_2), 'fastq_dict_2', [len(
            fastq_dict_2)])
    else:
        print('malloc', sys.getsizeof(fastq_dict_2), 'fastq_dict_2')
    with open(instrument_read(fastq_file_2, 'fastq_file_2'), 'r') as f_2:
        for line_2 in instrument_read(f_2, 'f_2'):
            print(18, 37)
            line_2 = instrument_read(line_2, 'line_2').strip()
            write_instrument_read(line_2, 'line_2')
            if type(line_2) == np.ndarray:
                print('malloc', sys.getsizeof(line_2), 'line_2', line_2.shape)
            elif type(line_2) == list:
                dims = []
                tmp = line_2
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(line_2), 'line_2', dims)
            elif type(line_2) == tuple:
                print('malloc', sys.getsizeof(line_2), 'line_2', [len(line_2)])
            else:
                print('malloc', sys.getsizeof(line_2), 'line_2')
            if not instrument_read(line_2, 'line_2'):
                continue
            if instrument_read(line_2, 'line_2').startswith('@'):
                print(22, 41)
                active_sequence_name_2 = instrument_read_sub(instrument_read
                    (line_2, 'line_2'), 'line_2', None, 1, None, True)
                write_instrument_read(active_sequence_name_2,
                    'active_sequence_name_2')
                if type(active_sequence_name_2) == np.ndarray:
                    print('malloc', sys.getsizeof(active_sequence_name_2),
                        'active_sequence_name_2', active_sequence_name_2.shape)
                elif type(active_sequence_name_2) == list:
                    dims = []
                    tmp = active_sequence_name_2
                    while type(tmp) == list:
                        dims.append(len(tmp))
                        if len(tmp) > 0:
                            tmp = tmp[0]
                        else:
                            tmp = None
                    print('malloc', sys.getsizeof(active_sequence_name_2),
                        'active_sequence_name_2', dims)
                elif type(active_sequence_name_2) == tuple:
                    print('malloc', sys.getsizeof(active_sequence_name_2),
                        'active_sequence_name_2', [len(active_sequence_name_2)]
                        )
                else:
                    print('malloc', sys.getsizeof(active_sequence_name_2),
                        'active_sequence_name_2')
                if instrument_read(active_sequence_name_2,
                    'active_sequence_name_2') not in instrument_read(
                    fastq_dict_2, 'fastq_dict_2'):
                    print(24, 43)
                    fastq_dict_2[instrument_read(instrument_read(
                        active_sequence_name_2, 'active_sequence_name_2'),
                        'active_sequence_name_2')] = []
                    write_instrument_read_sub(fastq_dict_2, 'fastq_dict_2',
                        instrument_read(instrument_read(
                        active_sequence_name_2, 'active_sequence_name_2'),
                        'active_sequence_name_2'), None, None, False)
                continue
            print(23, 45)
            sequence_2 = instrument_read(line_2, 'line_2')
            write_instrument_read(sequence_2, 'sequence_2')
            if type(sequence_2) == np.ndarray:
                print('malloc', sys.getsizeof(sequence_2), 'sequence_2',
                    sequence_2.shape)
            elif type(sequence_2) == list:
                dims = []
                tmp = sequence_2
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(sequence_2), 'sequence_2', dims)
            elif type(sequence_2) == tuple:
                print('malloc', sys.getsizeof(sequence_2), 'sequence_2', [
                    len(sequence_2)])
            else:
                print('malloc', sys.getsizeof(sequence_2), 'sequence_2')
            instrument_read_sub(instrument_read(fastq_dict_2,
                'fastq_dict_2'), 'fastq_dict_2', instrument_read(
                active_sequence_name_2, 'active_sequence_name_2'), None,
                None, False).append(instrument_read(sequence_2, 'sequence_2'))
    print('exit scope 2')
    return instrument_read(fastq_dict_2, 'fastq_dict_2')
    print('exit scope 2')


def read_fasta_with_quality(fasta_file, quality_file):
    print('enter scope 3')
    print(1, 50)
    print(29, 51)
    fasta_file_3 = instrument_read(fasta_file, 'fasta_file')
    write_instrument_read(fasta_file_3, 'fasta_file_3')
    if type(fasta_file_3) == np.ndarray:
        print('malloc', sys.getsizeof(fasta_file_3), 'fasta_file_3',
            fasta_file_3.shape)
    elif type(fasta_file_3) == list:
        dims = []
        tmp = fasta_file_3
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fasta_file_3), 'fasta_file_3', dims)
    elif type(fasta_file_3) == tuple:
        print('malloc', sys.getsizeof(fasta_file_3), 'fasta_file_3', [len(
            fasta_file_3)])
    else:
        print('malloc', sys.getsizeof(fasta_file_3), 'fasta_file_3')
    print(29, 52)
    quality_file_3 = instrument_read(quality_file, 'quality_file')
    write_instrument_read(quality_file_3, 'quality_file_3')
    if type(quality_file_3) == np.ndarray:
        print('malloc', sys.getsizeof(quality_file_3), 'quality_file_3',
            quality_file_3.shape)
    elif type(quality_file_3) == list:
        dims = []
        tmp = quality_file_3
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(quality_file_3), 'quality_file_3', dims)
    elif type(quality_file_3) == tuple:
        print('malloc', sys.getsizeof(quality_file_3), 'quality_file_3', [
            len(quality_file_3)])
    else:
        print('malloc', sys.getsizeof(quality_file_3), 'quality_file_3')
    """
    Reads a fasta file and a quality file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(29, 57)
    fasta_dict_3 = {}
    write_instrument_read(fasta_dict_3, 'fasta_dict_3')
    if type(fasta_dict_3) == np.ndarray:
        print('malloc', sys.getsizeof(fasta_dict_3), 'fasta_dict_3',
            fasta_dict_3.shape)
    elif type(fasta_dict_3) == list:
        dims = []
        tmp = fasta_dict_3
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fasta_dict_3), 'fasta_dict_3', dims)
    elif type(fasta_dict_3) == tuple:
        print('malloc', sys.getsizeof(fasta_dict_3), 'fasta_dict_3', [len(
            fasta_dict_3)])
    else:
        print('malloc', sys.getsizeof(fasta_dict_3), 'fasta_dict_3')
    with open(instrument_read(fasta_file_3, 'fasta_file_3'), 'r') as f_3:
        for line_3 in instrument_read(f_3, 'f_3'):
            print(31, 60)
            line_3 = instrument_read(line_3, 'line_3').strip()
            write_instrument_read(line_3, 'line_3')
            if type(line_3) == np.ndarray:
                print('malloc', sys.getsizeof(line_3), 'line_3', line_3.shape)
            elif type(line_3) == list:
                dims = []
                tmp = line_3
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(line_3), 'line_3', dims)
            elif type(line_3) == tuple:
                print('malloc', sys.getsizeof(line_3), 'line_3', [len(line_3)])
            else:
                print('malloc', sys.getsizeof(line_3), 'line_3')
            if not instrument_read(line_3, 'line_3'):
                continue
            if instrument_read(line_3, 'line_3').startswith('>'):
                print(35, 64)
                active_sequence_name_3 = instrument_read_sub(instrument_read
                    (line_3, 'line_3'), 'line_3', None, 1, None, True)
                write_instrument_read(active_sequence_name_3,
                    'active_sequence_name_3')
                if type(active_sequence_name_3) == np.ndarray:
                    print('malloc', sys.getsizeof(active_sequence_name_3),
                        'active_sequence_name_3', active_sequence_name_3.shape)
                elif type(active_sequence_name_3) == list:
                    dims = []
                    tmp = active_sequence_name_3
                    while type(tmp) == list:
                        dims.append(len(tmp))
                        if len(tmp) > 0:
                            tmp = tmp[0]
                        else:
                            tmp = None
                    print('malloc', sys.getsizeof(active_sequence_name_3),
                        'active_sequence_name_3', dims)
                elif type(active_sequence_name_3) == tuple:
                    print('malloc', sys.getsizeof(active_sequence_name_3),
                        'active_sequence_name_3', [len(active_sequence_name_3)]
                        )
                else:
                    print('malloc', sys.getsizeof(active_sequence_name_3),
                        'active_sequence_name_3')
                if instrument_read(active_sequence_name_3,
                    'active_sequence_name_3') not in instrument_read(
                    fasta_dict_3, 'fasta_dict_3'):
                    print(37, 66)
                    fasta_dict_3[instrument_read(instrument_read(
                        active_sequence_name_3, 'active_sequence_name_3'),
                        'active_sequence_name_3')] = []
                    write_instrument_read_sub(fasta_dict_3, 'fasta_dict_3',
                        instrument_read(instrument_read(
                        active_sequence_name_3, 'active_sequence_name_3'),
                        'active_sequence_name_3'), None, None, False)
                continue
            print(36, 68)
            sequence_3 = instrument_read(line_3, 'line_3')
            write_instrument_read(sequence_3, 'sequence_3')
            if type(sequence_3) == np.ndarray:
                print('malloc', sys.getsizeof(sequence_3), 'sequence_3',
                    sequence_3.shape)
            elif type(sequence_3) == list:
                dims = []
                tmp = sequence_3
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(sequence_3), 'sequence_3', dims)
            elif type(sequence_3) == tuple:
                print('malloc', sys.getsizeof(sequence_3), 'sequence_3', [
                    len(sequence_3)])
            else:
                print('malloc', sys.getsizeof(sequence_3), 'sequence_3')
            instrument_read_sub(instrument_read(fasta_dict_3,
                'fasta_dict_3'), 'fasta_dict_3', instrument_read(
                active_sequence_name_3, 'active_sequence_name_3'), None,
                None, False).append(instrument_read(sequence_3, 'sequence_3'))
    print(32, 70)
    quality_dict_3 = {}
    write_instrument_read(quality_dict_3, 'quality_dict_3')
    if type(quality_dict_3) == np.ndarray:
        print('malloc', sys.getsizeof(quality_dict_3), 'quality_dict_3',
            quality_dict_3.shape)
    elif type(quality_dict_3) == list:
        dims = []
        tmp = quality_dict_3
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(quality_dict_3), 'quality_dict_3', dims)
    elif type(quality_dict_3) == tuple:
        print('malloc', sys.getsizeof(quality_dict_3), 'quality_dict_3', [
            len(quality_dict_3)])
    else:
        print('malloc', sys.getsizeof(quality_dict_3), 'quality_dict_3')
    with open(instrument_read(quality_file_3, 'quality_file_3'), 'r') as f_3:
        for line_3 in instrument_read(f_3, 'f_3'):
            print(40, 73)
            line_3 = instrument_read(line_3, 'line_3').strip()
            write_instrument_read(line_3, 'line_3')
            if type(line_3) == np.ndarray:
                print('malloc', sys.getsizeof(line_3), 'line_3', line_3.shape)
            elif type(line_3) == list:
                dims = []
                tmp = line_3
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(line_3), 'line_3', dims)
            elif type(line_3) == tuple:
                print('malloc', sys.getsizeof(line_3), 'line_3', [len(line_3)])
            else:
                print('malloc', sys.getsizeof(line_3), 'line_3')
            if not instrument_read(line_3, 'line_3'):
                continue
            if instrument_read(line_3, 'line_3').startswith('>'):
                print(44, 77)
                active_sequence_name_3 = instrument_read_sub(instrument_read
                    (line_3, 'line_3'), 'line_3', None, 1, None, True)
                write_instrument_read(active_sequence_name_3,
                    'active_sequence_name_3')
                if type(active_sequence_name_3) == np.ndarray:
                    print('malloc', sys.getsizeof(active_sequence_name_3),
                        'active_sequence_name_3', active_sequence_name_3.shape)
                elif type(active_sequence_name_3) == list:
                    dims = []
                    tmp = active_sequence_name_3
                    while type(tmp) == list:
                        dims.append(len(tmp))
                        if len(tmp) > 0:
                            tmp = tmp[0]
                        else:
                            tmp = None
                    print('malloc', sys.getsizeof(active_sequence_name_3),
                        'active_sequence_name_3', dims)
                elif type(active_sequence_name_3) == tuple:
                    print('malloc', sys.getsizeof(active_sequence_name_3),
                        'active_sequence_name_3', [len(active_sequence_name_3)]
                        )
                else:
                    print('malloc', sys.getsizeof(active_sequence_name_3),
                        'active_sequence_name_3')
                if instrument_read(active_sequence_name_3,
                    'active_sequence_name_3') not in instrument_read(
                    quality_dict_3, 'quality_dict_3'):
                    print(46, 79)
                    quality_dict_3[instrument_read(instrument_read(
                        active_sequence_name_3, 'active_sequence_name_3'),
                        'active_sequence_name_3')] = []
                    write_instrument_read_sub(quality_dict_3,
                        'quality_dict_3', instrument_read(instrument_read(
                        active_sequence_name_3, 'active_sequence_name_3'),
                        'active_sequence_name_3'), None, None, False)
                continue
            print(45, 81)
            quality_3 = instrument_read(line_3, 'line_3')
            write_instrument_read(quality_3, 'quality_3')
            if type(quality_3) == np.ndarray:
                print('malloc', sys.getsizeof(quality_3), 'quality_3',
                    quality_3.shape)
            elif type(quality_3) == list:
                dims = []
                tmp = quality_3
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(quality_3), 'quality_3', dims)
            elif type(quality_3) == tuple:
                print('malloc', sys.getsizeof(quality_3), 'quality_3', [len
                    (quality_3)])
            else:
                print('malloc', sys.getsizeof(quality_3), 'quality_3')
            instrument_read_sub(instrument_read(quality_dict_3,
                'quality_dict_3'), 'quality_dict_3', instrument_read(
                active_sequence_name_3, 'active_sequence_name_3'), None,
                None, False).append(instrument_read(quality_3, 'quality_3'))
    print('exit scope 3')
    return instrument_read(fasta_dict_3, 'fasta_dict_3'), instrument_read(
        quality_dict_3, 'quality_dict_3')
    print('exit scope 3')


def darwin_wga_workflow(fasta_file, fastq_file, output_file, min_length=0,
    max_length=0, min_quality=0, max_quality=0, min_length_fraction=0,
    max_length_fraction=0):
    print('enter scope 4')
    print(1, 86)
    print(51, 89)
    fasta_file_4 = instrument_read(fasta_file, 'fasta_file')
    write_instrument_read(fasta_file_4, 'fasta_file_4')
    if type(fasta_file_4) == np.ndarray:
        print('malloc', sys.getsizeof(fasta_file_4), 'fasta_file_4',
            fasta_file_4.shape)
    elif type(fasta_file_4) == list:
        dims = []
        tmp = fasta_file_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fasta_file_4), 'fasta_file_4', dims)
    elif type(fasta_file_4) == tuple:
        print('malloc', sys.getsizeof(fasta_file_4), 'fasta_file_4', [len(
            fasta_file_4)])
    else:
        print('malloc', sys.getsizeof(fasta_file_4), 'fasta_file_4')
    print(51, 90)
    fastq_file_4 = instrument_read(fastq_file, 'fastq_file')
    write_instrument_read(fastq_file_4, 'fastq_file_4')
    if type(fastq_file_4) == np.ndarray:
        print('malloc', sys.getsizeof(fastq_file_4), 'fastq_file_4',
            fastq_file_4.shape)
    elif type(fastq_file_4) == list:
        dims = []
        tmp = fastq_file_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fastq_file_4), 'fastq_file_4', dims)
    elif type(fastq_file_4) == tuple:
        print('malloc', sys.getsizeof(fastq_file_4), 'fastq_file_4', [len(
            fastq_file_4)])
    else:
        print('malloc', sys.getsizeof(fastq_file_4), 'fastq_file_4')
    print(51, 91)
    output_file_4 = instrument_read(output_file, 'output_file')
    write_instrument_read(output_file_4, 'output_file_4')
    if type(output_file_4) == np.ndarray:
        print('malloc', sys.getsizeof(output_file_4), 'output_file_4',
            output_file_4.shape)
    elif type(output_file_4) == list:
        dims = []
        tmp = output_file_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(output_file_4), 'output_file_4', dims)
    elif type(output_file_4) == tuple:
        print('malloc', sys.getsizeof(output_file_4), 'output_file_4', [len
            (output_file_4)])
    else:
        print('malloc', sys.getsizeof(output_file_4), 'output_file_4')
    print(51, 92)
    min_length_4 = instrument_read(min_length, 'min_length')
    write_instrument_read(min_length_4, 'min_length_4')
    if type(min_length_4) == np.ndarray:
        print('malloc', sys.getsizeof(min_length_4), 'min_length_4',
            min_length_4.shape)
    elif type(min_length_4) == list:
        dims = []
        tmp = min_length_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(min_length_4), 'min_length_4', dims)
    elif type(min_length_4) == tuple:
        print('malloc', sys.getsizeof(min_length_4), 'min_length_4', [len(
            min_length_4)])
    else:
        print('malloc', sys.getsizeof(min_length_4), 'min_length_4')
    print(51, 93)
    max_length_4 = instrument_read(max_length, 'max_length')
    write_instrument_read(max_length_4, 'max_length_4')
    if type(max_length_4) == np.ndarray:
        print('malloc', sys.getsizeof(max_length_4), 'max_length_4',
            max_length_4.shape)
    elif type(max_length_4) == list:
        dims = []
        tmp = max_length_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(max_length_4), 'max_length_4', dims)
    elif type(max_length_4) == tuple:
        print('malloc', sys.getsizeof(max_length_4), 'max_length_4', [len(
            max_length_4)])
    else:
        print('malloc', sys.getsizeof(max_length_4), 'max_length_4')
    print(51, 94)
    min_quality_4 = instrument_read(min_quality, 'min_quality')
    write_instrument_read(min_quality_4, 'min_quality_4')
    if type(min_quality_4) == np.ndarray:
        print('malloc', sys.getsizeof(min_quality_4), 'min_quality_4',
            min_quality_4.shape)
    elif type(min_quality_4) == list:
        dims = []
        tmp = min_quality_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(min_quality_4), 'min_quality_4', dims)
    elif type(min_quality_4) == tuple:
        print('malloc', sys.getsizeof(min_quality_4), 'min_quality_4', [len
            (min_quality_4)])
    else:
        print('malloc', sys.getsizeof(min_quality_4), 'min_quality_4')
    print(51, 95)
    max_quality_4 = instrument_read(max_quality, 'max_quality')
    write_instrument_read(max_quality_4, 'max_quality_4')
    if type(max_quality_4) == np.ndarray:
        print('malloc', sys.getsizeof(max_quality_4), 'max_quality_4',
            max_quality_4.shape)
    elif type(max_quality_4) == list:
        dims = []
        tmp = max_quality_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(max_quality_4), 'max_quality_4', dims)
    elif type(max_quality_4) == tuple:
        print('malloc', sys.getsizeof(max_quality_4), 'max_quality_4', [len
            (max_quality_4)])
    else:
        print('malloc', sys.getsizeof(max_quality_4), 'max_quality_4')
    print(51, 96)
    min_length_fraction_4 = instrument_read(min_length_fraction,
        'min_length_fraction')
    write_instrument_read(min_length_fraction_4, 'min_length_fraction_4')
    if type(min_length_fraction_4) == np.ndarray:
        print('malloc', sys.getsizeof(min_length_fraction_4),
            'min_length_fraction_4', min_length_fraction_4.shape)
    elif type(min_length_fraction_4) == list:
        dims = []
        tmp = min_length_fraction_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(min_length_fraction_4),
            'min_length_fraction_4', dims)
    elif type(min_length_fraction_4) == tuple:
        print('malloc', sys.getsizeof(min_length_fraction_4),
            'min_length_fraction_4', [len(min_length_fraction_4)])
    else:
        print('malloc', sys.getsizeof(min_length_fraction_4),
            'min_length_fraction_4')
    print(51, 97)
    max_length_fraction_4 = instrument_read(max_length_fraction,
        'max_length_fraction')
    write_instrument_read(max_length_fraction_4, 'max_length_fraction_4')
    if type(max_length_fraction_4) == np.ndarray:
        print('malloc', sys.getsizeof(max_length_fraction_4),
            'max_length_fraction_4', max_length_fraction_4.shape)
    elif type(max_length_fraction_4) == list:
        dims = []
        tmp = max_length_fraction_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(max_length_fraction_4),
            'max_length_fraction_4', dims)
    elif type(max_length_fraction_4) == tuple:
        print('malloc', sys.getsizeof(max_length_fraction_4),
            'max_length_fraction_4', [len(max_length_fraction_4)])
    else:
        print('malloc', sys.getsizeof(max_length_fraction_4),
            'max_length_fraction_4')
    """
    DARWIN-Whole Genome Alignment workflow
    """
    print(51, 101)
    fasta_dict_4 = read_fasta(instrument_read(fasta_file_4, 'fasta_file_4'))
    write_instrument_read(fasta_dict_4, 'fasta_dict_4')
    if type(fasta_dict_4) == np.ndarray:
        print('malloc', sys.getsizeof(fasta_dict_4), 'fasta_dict_4',
            fasta_dict_4.shape)
    elif type(fasta_dict_4) == list:
        dims = []
        tmp = fasta_dict_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fasta_dict_4), 'fasta_dict_4', dims)
    elif type(fasta_dict_4) == tuple:
        print('malloc', sys.getsizeof(fasta_dict_4), 'fasta_dict_4', [len(
            fasta_dict_4)])
    else:
        print('malloc', sys.getsizeof(fasta_dict_4), 'fasta_dict_4')
    print(51, 102)
    fastq_dict_4 = read_fastq(instrument_read(fastq_file_4, 'fastq_file_4'))
    write_instrument_read(fastq_dict_4, 'fastq_dict_4')
    if type(fastq_dict_4) == np.ndarray:
        print('malloc', sys.getsizeof(fastq_dict_4), 'fastq_dict_4',
            fastq_dict_4.shape)
    elif type(fastq_dict_4) == list:
        dims = []
        tmp = fastq_dict_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fastq_dict_4), 'fastq_dict_4', dims)
    elif type(fastq_dict_4) == tuple:
        print('malloc', sys.getsizeof(fastq_dict_4), 'fastq_dict_4', [len(
            fastq_dict_4)])
    else:
        print('malloc', sys.getsizeof(fastq_dict_4), 'fastq_dict_4')
    print(51, 103)
    filtered_fasta_dict_4 = filter_fasta(instrument_read(fasta_dict_4,
        'fasta_dict_4'), instrument_read(min_length_4, 'min_length_4'),
        instrument_read(max_length_4, 'max_length_4'), instrument_read(
        min_length_fraction_4, 'min_length_fraction_4'), instrument_read(
        max_length_fraction_4, 'max_length_fraction_4'))
    write_instrument_read(filtered_fasta_dict_4, 'filtered_fasta_dict_4')
    if type(filtered_fasta_dict_4) == np.ndarray:
        print('malloc', sys.getsizeof(filtered_fasta_dict_4),
            'filtered_fasta_dict_4', filtered_fasta_dict_4.shape)
    elif type(filtered_fasta_dict_4) == list:
        dims = []
        tmp = filtered_fasta_dict_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(filtered_fasta_dict_4),
            'filtered_fasta_dict_4', dims)
    elif type(filtered_fasta_dict_4) == tuple:
        print('malloc', sys.getsizeof(filtered_fasta_dict_4),
            'filtered_fasta_dict_4', [len(filtered_fasta_dict_4)])
    else:
        print('malloc', sys.getsizeof(filtered_fasta_dict_4),
            'filtered_fasta_dict_4')
    print(51, 105)
    filtered_fastq_dict_4 = filter_fastq(instrument_read(fastq_dict_4,
        'fastq_dict_4'), instrument_read(min_quality_4, 'min_quality_4'),
        instrument_read(max_quality_4, 'max_quality_4'))
    write_instrument_read(filtered_fastq_dict_4, 'filtered_fastq_dict_4')
    if type(filtered_fastq_dict_4) == np.ndarray:
        print('malloc', sys.getsizeof(filtered_fastq_dict_4),
            'filtered_fastq_dict_4', filtered_fastq_dict_4.shape)
    elif type(filtered_fastq_dict_4) == list:
        dims = []
        tmp = filtered_fastq_dict_4
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(filtered_fastq_dict_4),
            'filtered_fastq_dict_4', dims)
    elif type(filtered_fastq_dict_4) == tuple:
        print('malloc', sys.getsizeof(filtered_fastq_dict_4),
            'filtered_fastq_dict_4', [len(filtered_fastq_dict_4)])
    else:
        print('malloc', sys.getsizeof(filtered_fastq_dict_4),
            'filtered_fastq_dict_4')
    write_output(instrument_read(filtered_fasta_dict_4,
        'filtered_fasta_dict_4'), instrument_read(filtered_fastq_dict_4,
        'filtered_fastq_dict_4'), instrument_read(output_file_4,
        'output_file_4'))
    print('exit scope 4')


def filter_fasta(fasta_dict, min_length, max_length, min_length_fraction,
    max_length_fraction):
    print('enter scope 5')
    print(1, 110)
    print(54, 112)
    fasta_dict_5 = instrument_read(fasta_dict, 'fasta_dict')
    write_instrument_read(fasta_dict_5, 'fasta_dict_5')
    if type(fasta_dict_5) == np.ndarray:
        print('malloc', sys.getsizeof(fasta_dict_5), 'fasta_dict_5',
            fasta_dict_5.shape)
    elif type(fasta_dict_5) == list:
        dims = []
        tmp = fasta_dict_5
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fasta_dict_5), 'fasta_dict_5', dims)
    elif type(fasta_dict_5) == tuple:
        print('malloc', sys.getsizeof(fasta_dict_5), 'fasta_dict_5', [len(
            fasta_dict_5)])
    else:
        print('malloc', sys.getsizeof(fasta_dict_5), 'fasta_dict_5')
    print(54, 113)
    min_length_5 = instrument_read(min_length, 'min_length')
    write_instrument_read(min_length_5, 'min_length_5')
    if type(min_length_5) == np.ndarray:
        print('malloc', sys.getsizeof(min_length_5), 'min_length_5',
            min_length_5.shape)
    elif type(min_length_5) == list:
        dims = []
        tmp = min_length_5
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(min_length_5), 'min_length_5', dims)
    elif type(min_length_5) == tuple:
        print('malloc', sys.getsizeof(min_length_5), 'min_length_5', [len(
            min_length_5)])
    else:
        print('malloc', sys.getsizeof(min_length_5), 'min_length_5')
    print(54, 114)
    max_length_5 = instrument_read(max_length, 'max_length')
    write_instrument_read(max_length_5, 'max_length_5')
    if type(max_length_5) == np.ndarray:
        print('malloc', sys.getsizeof(max_length_5), 'max_length_5',
            max_length_5.shape)
    elif type(max_length_5) == list:
        dims = []
        tmp = max_length_5
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(max_length_5), 'max_length_5', dims)
    elif type(max_length_5) == tuple:
        print('malloc', sys.getsizeof(max_length_5), 'max_length_5', [len(
            max_length_5)])
    else:
        print('malloc', sys.getsizeof(max_length_5), 'max_length_5')
    print(54, 115)
    min_length_fraction_5 = instrument_read(min_length_fraction,
        'min_length_fraction')
    write_instrument_read(min_length_fraction_5, 'min_length_fraction_5')
    if type(min_length_fraction_5) == np.ndarray:
        print('malloc', sys.getsizeof(min_length_fraction_5),
            'min_length_fraction_5', min_length_fraction_5.shape)
    elif type(min_length_fraction_5) == list:
        dims = []
        tmp = min_length_fraction_5
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(min_length_fraction_5),
            'min_length_fraction_5', dims)
    elif type(min_length_fraction_5) == tuple:
        print('malloc', sys.getsizeof(min_length_fraction_5),
            'min_length_fraction_5', [len(min_length_fraction_5)])
    else:
        print('malloc', sys.getsizeof(min_length_fraction_5),
            'min_length_fraction_5')
    print(54, 116)
    max_length_fraction_5 = instrument_read(max_length_fraction,
        'max_length_fraction')
    write_instrument_read(max_length_fraction_5, 'max_length_fraction_5')
    if type(max_length_fraction_5) == np.ndarray:
        print('malloc', sys.getsizeof(max_length_fraction_5),
            'max_length_fraction_5', max_length_fraction_5.shape)
    elif type(max_length_fraction_5) == list:
        dims = []
        tmp = max_length_fraction_5
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(max_length_fraction_5),
            'max_length_fraction_5', dims)
    elif type(max_length_fraction_5) == tuple:
        print('malloc', sys.getsizeof(max_length_fraction_5),
            'max_length_fraction_5', [len(max_length_fraction_5)])
    else:
        print('malloc', sys.getsizeof(max_length_fraction_5),
            'max_length_fraction_5')
    """
    Filters a fasta dictionary by length
    """
    print(54, 120)
    filtered_fasta_dict_5 = {}
    write_instrument_read(filtered_fasta_dict_5, 'filtered_fasta_dict_5')
    if type(filtered_fasta_dict_5) == np.ndarray:
        print('malloc', sys.getsizeof(filtered_fasta_dict_5),
            'filtered_fasta_dict_5', filtered_fasta_dict_5.shape)
    elif type(filtered_fasta_dict_5) == list:
        dims = []
        tmp = filtered_fasta_dict_5
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(filtered_fasta_dict_5),
            'filtered_fasta_dict_5', dims)
    elif type(filtered_fasta_dict_5) == tuple:
        print('malloc', sys.getsizeof(filtered_fasta_dict_5),
            'filtered_fasta_dict_5', [len(filtered_fasta_dict_5)])
    else:
        print('malloc', sys.getsizeof(filtered_fasta_dict_5),
            'filtered_fasta_dict_5')
    for key_5 in instrument_read(fasta_dict_5, 'fasta_dict_5'):
        print(56, 122)
        sequence_5 = instrument_read_sub(instrument_read_sub(
            instrument_read(fasta_dict_5, 'fasta_dict_5'), 'fasta_dict_5',
            instrument_read(key_5, 'key_5'), None, None, False),
            'fasta_dict_5[key_5]', 0, None, None, False)
        write_instrument_read(sequence_5, 'sequence_5')
        if type(sequence_5) == np.ndarray:
            print('malloc', sys.getsizeof(sequence_5), 'sequence_5',
                sequence_5.shape)
        elif type(sequence_5) == list:
            dims = []
            tmp = sequence_5
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(sequence_5), 'sequence_5', dims)
        elif type(sequence_5) == tuple:
            print('malloc', sys.getsizeof(sequence_5), 'sequence_5', [len(
                sequence_5)])
        else:
            print('malloc', sys.getsizeof(sequence_5), 'sequence_5')
        print(56, 123)
        length_5 = len(instrument_read(sequence_5, 'sequence_5'))
        write_instrument_read(length_5, 'length_5')
        if type(length_5) == np.ndarray:
            print('malloc', sys.getsizeof(length_5), 'length_5', length_5.shape
                )
        elif type(length_5) == list:
            dims = []
            tmp = length_5
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(length_5), 'length_5', dims)
        elif type(length_5) == tuple:
            print('malloc', sys.getsizeof(length_5), 'length_5', [len(
                length_5)])
        else:
            print('malloc', sys.getsizeof(length_5), 'length_5')
        if instrument_read(min_length_fraction_5, 'min_length_fraction_5') > 0:
            if instrument_read(length_5, 'length_5') < instrument_read(
                min_length_fraction_5, 'min_length_fraction_5') * len(
                instrument_read_sub(instrument_read_sub(instrument_read(
                fasta_dict_5, 'fasta_dict_5'), 'fasta_dict_5',
                instrument_read(key_5, 'key_5'), None, None, False),
                'fasta_dict_5[key_5]', 0, None, None, False)):
                continue
        if instrument_read(max_length_fraction_5, 'max_length_fraction_5') > 0:
            if instrument_read(length_5, 'length_5') > instrument_read(
                max_length_fraction_5, 'max_length_fraction_5') * len(
                instrument_read_sub(instrument_read_sub(instrument_read(
                fasta_dict_5, 'fasta_dict_5'), 'fasta_dict_5',
                instrument_read(key_5, 'key_5'), None, None, False),
                'fasta_dict_5[key_5]', 0, None, None, False)):
                continue
        if instrument_read(min_length_5, 'min_length_5') > 0:
            if instrument_read(length_5, 'length_5') < instrument_read(
                min_length_5, 'min_length_5'):
                continue
        if instrument_read(max_length_5, 'max_length_5') > 0:
            if instrument_read(length_5, 'length_5') > instrument_read(
                max_length_5, 'max_length_5'):
                continue
        print(71, 136)
        filtered_fasta_dict_5[instrument_read(instrument_read(key_5,
            'key_5'), 'key_5')] = instrument_read_sub(instrument_read(
            fasta_dict_5, 'fasta_dict_5'), 'fasta_dict_5', instrument_read(
            key_5, 'key_5'), None, None, False)
        write_instrument_read_sub(filtered_fasta_dict_5,
            'filtered_fasta_dict_5', instrument_read(instrument_read(key_5,
            'key_5'), 'key_5'), None, None, False)
    print('exit scope 5')
    return instrument_read(filtered_fasta_dict_5, 'filtered_fasta_dict_5')
    print('exit scope 5')


def filter_fastq(fastq_dict, min_quality, max_quality):
    print('enter scope 6')
    print(1, 140)
    print(77, 141)
    fastq_dict_6 = instrument_read(fastq_dict, 'fastq_dict')
    write_instrument_read(fastq_dict_6, 'fastq_dict_6')
    if type(fastq_dict_6) == np.ndarray:
        print('malloc', sys.getsizeof(fastq_dict_6), 'fastq_dict_6',
            fastq_dict_6.shape)
    elif type(fastq_dict_6) == list:
        dims = []
        tmp = fastq_dict_6
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fastq_dict_6), 'fastq_dict_6', dims)
    elif type(fastq_dict_6) == tuple:
        print('malloc', sys.getsizeof(fastq_dict_6), 'fastq_dict_6', [len(
            fastq_dict_6)])
    else:
        print('malloc', sys.getsizeof(fastq_dict_6), 'fastq_dict_6')
    print(77, 142)
    min_quality_6 = instrument_read(min_quality, 'min_quality')
    write_instrument_read(min_quality_6, 'min_quality_6')
    if type(min_quality_6) == np.ndarray:
        print('malloc', sys.getsizeof(min_quality_6), 'min_quality_6',
            min_quality_6.shape)
    elif type(min_quality_6) == list:
        dims = []
        tmp = min_quality_6
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(min_quality_6), 'min_quality_6', dims)
    elif type(min_quality_6) == tuple:
        print('malloc', sys.getsizeof(min_quality_6), 'min_quality_6', [len
            (min_quality_6)])
    else:
        print('malloc', sys.getsizeof(min_quality_6), 'min_quality_6')
    print(77, 143)
    max_quality_6 = instrument_read(max_quality, 'max_quality')
    write_instrument_read(max_quality_6, 'max_quality_6')
    if type(max_quality_6) == np.ndarray:
        print('malloc', sys.getsizeof(max_quality_6), 'max_quality_6',
            max_quality_6.shape)
    elif type(max_quality_6) == list:
        dims = []
        tmp = max_quality_6
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(max_quality_6), 'max_quality_6', dims)
    elif type(max_quality_6) == tuple:
        print('malloc', sys.getsizeof(max_quality_6), 'max_quality_6', [len
            (max_quality_6)])
    else:
        print('malloc', sys.getsizeof(max_quality_6), 'max_quality_6')
    """
    Filters a fastq dictionary by quality
    """
    print(77, 147)
    filtered_fastq_dict_6 = {}
    write_instrument_read(filtered_fastq_dict_6, 'filtered_fastq_dict_6')
    if type(filtered_fastq_dict_6) == np.ndarray:
        print('malloc', sys.getsizeof(filtered_fastq_dict_6),
            'filtered_fastq_dict_6', filtered_fastq_dict_6.shape)
    elif type(filtered_fastq_dict_6) == list:
        dims = []
        tmp = filtered_fastq_dict_6
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(filtered_fastq_dict_6),
            'filtered_fastq_dict_6', dims)
    elif type(filtered_fastq_dict_6) == tuple:
        print('malloc', sys.getsizeof(filtered_fastq_dict_6),
            'filtered_fastq_dict_6', [len(filtered_fastq_dict_6)])
    else:
        print('malloc', sys.getsizeof(filtered_fastq_dict_6),
            'filtered_fastq_dict_6')
    for key_6 in instrument_read(fastq_dict_6, 'fastq_dict_6'):
        print(79, 149)
        quality_6 = instrument_read_sub(instrument_read_sub(instrument_read
            (fastq_dict_6, 'fastq_dict_6'), 'fastq_dict_6', instrument_read
            (key_6, 'key_6'), None, None, False), 'fastq_dict_6[key_6]', 0,
            None, None, False)
        write_instrument_read(quality_6, 'quality_6')
        if type(quality_6) == np.ndarray:
            print('malloc', sys.getsizeof(quality_6), 'quality_6',
                quality_6.shape)
        elif type(quality_6) == list:
            dims = []
            tmp = quality_6
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(quality_6), 'quality_6', dims)
        elif type(quality_6) == tuple:
            print('malloc', sys.getsizeof(quality_6), 'quality_6', [len(
                quality_6)])
        else:
            print('malloc', sys.getsizeof(quality_6), 'quality_6')
        if instrument_read(min_quality_6, 'min_quality_6') > 0:
            if instrument_read(quality_6, 'quality_6') < instrument_read(
                min_quality_6, 'min_quality_6'):
                continue
        if instrument_read(max_quality_6, 'max_quality_6') > 0:
            if instrument_read(quality_6, 'quality_6') > instrument_read(
                max_quality_6, 'max_quality_6'):
                continue
        print(86, 156)
        filtered_fastq_dict_6[instrument_read(instrument_read(key_6,
            'key_6'), 'key_6')] = instrument_read_sub(instrument_read(
            fastq_dict_6, 'fastq_dict_6'), 'fastq_dict_6', instrument_read(
            key_6, 'key_6'), None, None, False)
        write_instrument_read_sub(filtered_fastq_dict_6,
            'filtered_fastq_dict_6', instrument_read(instrument_read(key_6,
            'key_6'), 'key_6'), None, None, False)
    print('exit scope 6')
    return instrument_read(filtered_fastq_dict_6, 'filtered_fastq_dict_6')
    print('exit scope 6')


def write_output(filtered_fasta_dict, filtered_fastq_dict, output_file):
    print('enter scope 7')
    print(1, 160)
    print(92, 161)
    filtered_fasta_dict_7 = instrument_read(filtered_fasta_dict,
        'filtered_fasta_dict')
    write_instrument_read(filtered_fasta_dict_7, 'filtered_fasta_dict_7')
    if type(filtered_fasta_dict_7) == np.ndarray:
        print('malloc', sys.getsizeof(filtered_fasta_dict_7),
            'filtered_fasta_dict_7', filtered_fasta_dict_7.shape)
    elif type(filtered_fasta_dict_7) == list:
        dims = []
        tmp = filtered_fasta_dict_7
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(filtered_fasta_dict_7),
            'filtered_fasta_dict_7', dims)
    elif type(filtered_fasta_dict_7) == tuple:
        print('malloc', sys.getsizeof(filtered_fasta_dict_7),
            'filtered_fasta_dict_7', [len(filtered_fasta_dict_7)])
    else:
        print('malloc', sys.getsizeof(filtered_fasta_dict_7),
            'filtered_fasta_dict_7')
    print(92, 162)
    filtered_fastq_dict_7 = instrument_read(filtered_fastq_dict,
        'filtered_fastq_dict')
    write_instrument_read(filtered_fastq_dict_7, 'filtered_fastq_dict_7')
    if type(filtered_fastq_dict_7) == np.ndarray:
        print('malloc', sys.getsizeof(filtered_fastq_dict_7),
            'filtered_fastq_dict_7', filtered_fastq_dict_7.shape)
    elif type(filtered_fastq_dict_7) == list:
        dims = []
        tmp = filtered_fastq_dict_7
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(filtered_fastq_dict_7),
            'filtered_fastq_dict_7', dims)
    elif type(filtered_fastq_dict_7) == tuple:
        print('malloc', sys.getsizeof(filtered_fastq_dict_7),
            'filtered_fastq_dict_7', [len(filtered_fastq_dict_7)])
    else:
        print('malloc', sys.getsizeof(filtered_fastq_dict_7),
            'filtered_fastq_dict_7')
    print(92, 163)
    output_file_7 = instrument_read(output_file, 'output_file')
    write_instrument_read(output_file_7, 'output_file_7')
    if type(output_file_7) == np.ndarray:
        print('malloc', sys.getsizeof(output_file_7), 'output_file_7',
            output_file_7.shape)
    elif type(output_file_7) == list:
        dims = []
        tmp = output_file_7
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(output_file_7), 'output_file_7', dims)
    elif type(output_file_7) == tuple:
        print('malloc', sys.getsizeof(output_file_7), 'output_file_7', [len
            (output_file_7)])
    else:
        print('malloc', sys.getsizeof(output_file_7), 'output_file_7')
    """
    Writes output
    """
    with open(instrument_read(output_file_7, 'output_file_7'), 'w') as f_7:
        for key_7 in instrument_read(filtered_fasta_dict_7,
            'filtered_fasta_dict_7'):
            instrument_read(f_7, 'f_7').write('>' + instrument_read(key_7,
                'key_7') + '\n')
            instrument_read(f_7, 'f_7').write(instrument_read_sub(
                instrument_read_sub(instrument_read(filtered_fasta_dict_7,
                'filtered_fasta_dict_7'), 'filtered_fasta_dict_7',
                instrument_read(key_7, 'key_7'), None, None, False),
                'filtered_fasta_dict_7[key_7]', 0, None, None, False) + '\n')
        for key_7 in instrument_read(filtered_fastq_dict_7,
            'filtered_fastq_dict_7'):
            instrument_read(f_7, 'f_7').write('@' + instrument_read(key_7,
                'key_7') + '\n')
            instrument_read(f_7, 'f_7').write(instrument_read_sub(
                instrument_read_sub(instrument_read(filtered_fastq_dict_7,
                'filtered_fastq_dict_7'), 'filtered_fastq_dict_7',
                instrument_read(key_7, 'key_7'), None, None, False),
                'filtered_fastq_dict_7[key_7]', 0, None, None, False) + '\n')
            instrument_read(f_7, 'f_7').write('+\n')
            instrument_read(f_7, 'f_7').write(instrument_read_sub(
                instrument_read_sub(instrument_read(filtered_fastq_dict_7,
                'filtered_fastq_dict_7'), 'filtered_fastq_dict_7',
                instrument_read(key_7, 'key_7'), None, None, False),
                'filtered_fastq_dict_7[key_7]', 1, None, None, False) + '\n')
    print('exit scope 7')


def smith_waterman(seq1, seq2, match_score=3, mismatch_score=-3, gap_score=-2):
    print('enter scope 8')
    print(1, 178)
    print(100, 179)
    seq1_8 = instrument_read(seq1, 'seq1')
    write_instrument_read(seq1_8, 'seq1_8')
    if type(seq1_8) == np.ndarray:
        print('malloc', sys.getsizeof(seq1_8), 'seq1_8', seq1_8.shape)
    elif type(seq1_8) == list:
        dims = []
        tmp = seq1_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(seq1_8), 'seq1_8', dims)
    elif type(seq1_8) == tuple:
        print('malloc', sys.getsizeof(seq1_8), 'seq1_8', [len(seq1_8)])
    else:
        print('malloc', sys.getsizeof(seq1_8), 'seq1_8')
    print(100, 180)
    seq2_8 = instrument_read(seq2, 'seq2')
    write_instrument_read(seq2_8, 'seq2_8')
    if type(seq2_8) == np.ndarray:
        print('malloc', sys.getsizeof(seq2_8), 'seq2_8', seq2_8.shape)
    elif type(seq2_8) == list:
        dims = []
        tmp = seq2_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(seq2_8), 'seq2_8', dims)
    elif type(seq2_8) == tuple:
        print('malloc', sys.getsizeof(seq2_8), 'seq2_8', [len(seq2_8)])
    else:
        print('malloc', sys.getsizeof(seq2_8), 'seq2_8')
    print(100, 181)
    match_score_8 = instrument_read(match_score, 'match_score')
    write_instrument_read(match_score_8, 'match_score_8')
    if type(match_score_8) == np.ndarray:
        print('malloc', sys.getsizeof(match_score_8), 'match_score_8',
            match_score_8.shape)
    elif type(match_score_8) == list:
        dims = []
        tmp = match_score_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(match_score_8), 'match_score_8', dims)
    elif type(match_score_8) == tuple:
        print('malloc', sys.getsizeof(match_score_8), 'match_score_8', [len
            (match_score_8)])
    else:
        print('malloc', sys.getsizeof(match_score_8), 'match_score_8')
    print(100, 182)
    mismatch_score_8 = instrument_read(mismatch_score, 'mismatch_score')
    write_instrument_read(mismatch_score_8, 'mismatch_score_8')
    if type(mismatch_score_8) == np.ndarray:
        print('malloc', sys.getsizeof(mismatch_score_8), 'mismatch_score_8',
            mismatch_score_8.shape)
    elif type(mismatch_score_8) == list:
        dims = []
        tmp = mismatch_score_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(mismatch_score_8), 'mismatch_score_8',
            dims)
    elif type(mismatch_score_8) == tuple:
        print('malloc', sys.getsizeof(mismatch_score_8), 'mismatch_score_8',
            [len(mismatch_score_8)])
    else:
        print('malloc', sys.getsizeof(mismatch_score_8), 'mismatch_score_8')
    print(100, 183)
    gap_score_8 = instrument_read(gap_score, 'gap_score')
    write_instrument_read(gap_score_8, 'gap_score_8')
    if type(gap_score_8) == np.ndarray:
        print('malloc', sys.getsizeof(gap_score_8), 'gap_score_8',
            gap_score_8.shape)
    elif type(gap_score_8) == list:
        dims = []
        tmp = gap_score_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(gap_score_8), 'gap_score_8', dims)
    elif type(gap_score_8) == tuple:
        print('malloc', sys.getsizeof(gap_score_8), 'gap_score_8', [len(
            gap_score_8)])
    else:
        print('malloc', sys.getsizeof(gap_score_8), 'gap_score_8')
    """
    Computes the local alignment between two sequences using a scoring matrix.
    """
    print(100, 187)
    scoring_matrix_8 = [[(0) for j_8 in range(len(instrument_read(seq2_8,
        'seq2_8')) + 1)] for i_8 in range(len(instrument_read(seq1_8,
        'seq1_8')) + 1)]
    write_instrument_read(scoring_matrix_8, 'scoring_matrix_8')
    if type(scoring_matrix_8) == np.ndarray:
        print('malloc', sys.getsizeof(scoring_matrix_8), 'scoring_matrix_8',
            scoring_matrix_8.shape)
    elif type(scoring_matrix_8) == list:
        dims = []
        tmp = scoring_matrix_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(scoring_matrix_8), 'scoring_matrix_8',
            dims)
    elif type(scoring_matrix_8) == tuple:
        print('malloc', sys.getsizeof(scoring_matrix_8), 'scoring_matrix_8',
            [len(scoring_matrix_8)])
    else:
        print('malloc', sys.getsizeof(scoring_matrix_8), 'scoring_matrix_8')
    print(100, 189)
    traceback_matrix_8 = [[None for j_8 in range(len(instrument_read(seq2_8,
        'seq2_8')) + 1)] for i_8 in range(len(instrument_read(seq1_8,
        'seq1_8')) + 1)]
    write_instrument_read(traceback_matrix_8, 'traceback_matrix_8')
    if type(traceback_matrix_8) == np.ndarray:
        print('malloc', sys.getsizeof(traceback_matrix_8),
            'traceback_matrix_8', traceback_matrix_8.shape)
    elif type(traceback_matrix_8) == list:
        dims = []
        tmp = traceback_matrix_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(traceback_matrix_8),
            'traceback_matrix_8', dims)
    elif type(traceback_matrix_8) == tuple:
        print('malloc', sys.getsizeof(traceback_matrix_8),
            'traceback_matrix_8', [len(traceback_matrix_8)])
    else:
        print('malloc', sys.getsizeof(traceback_matrix_8), 'traceback_matrix_8'
            )
    print(100, 191)
    max_score_8 = 0
    write_instrument_read(max_score_8, 'max_score_8')
    if type(max_score_8) == np.ndarray:
        print('malloc', sys.getsizeof(max_score_8), 'max_score_8',
            max_score_8.shape)
    elif type(max_score_8) == list:
        dims = []
        tmp = max_score_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(max_score_8), 'max_score_8', dims)
    elif type(max_score_8) == tuple:
        print('malloc', sys.getsizeof(max_score_8), 'max_score_8', [len(
            max_score_8)])
    else:
        print('malloc', sys.getsizeof(max_score_8), 'max_score_8')
    print(100, 192)
    max_pos_8 = None
    write_instrument_read(max_pos_8, 'max_pos_8')
    if type(max_pos_8) == np.ndarray:
        print('malloc', sys.getsizeof(max_pos_8), 'max_pos_8', max_pos_8.shape)
    elif type(max_pos_8) == list:
        dims = []
        tmp = max_pos_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(max_pos_8), 'max_pos_8', dims)
    elif type(max_pos_8) == tuple:
        print('malloc', sys.getsizeof(max_pos_8), 'max_pos_8', [len(max_pos_8)]
            )
    else:
        print('malloc', sys.getsizeof(max_pos_8), 'max_pos_8')
    for i_8 in range(1, len(instrument_read(seq1_8, 'seq1_8')) + 1):
        for j_8 in range(1, len(instrument_read(seq2_8, 'seq2_8')) + 1):
            print(104, 195)
            letter1_8 = instrument_read_sub(instrument_read(seq1_8,
                'seq1_8'), 'seq1_8', instrument_read(i_8, 'i_8') - 1, None,
                None, False)
            write_instrument_read(letter1_8, 'letter1_8')
            if type(letter1_8) == np.ndarray:
                print('malloc', sys.getsizeof(letter1_8), 'letter1_8',
                    letter1_8.shape)
            elif type(letter1_8) == list:
                dims = []
                tmp = letter1_8
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(letter1_8), 'letter1_8', dims)
            elif type(letter1_8) == tuple:
                print('malloc', sys.getsizeof(letter1_8), 'letter1_8', [len
                    (letter1_8)])
            else:
                print('malloc', sys.getsizeof(letter1_8), 'letter1_8')
            print(104, 196)
            letter2_8 = instrument_read_sub(instrument_read(seq2_8,
                'seq2_8'), 'seq2_8', instrument_read(j_8, 'j_8') - 1, None,
                None, False)
            write_instrument_read(letter2_8, 'letter2_8')
            if type(letter2_8) == np.ndarray:
                print('malloc', sys.getsizeof(letter2_8), 'letter2_8',
                    letter2_8.shape)
            elif type(letter2_8) == list:
                dims = []
                tmp = letter2_8
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(letter2_8), 'letter2_8', dims)
            elif type(letter2_8) == tuple:
                print('malloc', sys.getsizeof(letter2_8), 'letter2_8', [len
                    (letter2_8)])
            else:
                print('malloc', sys.getsizeof(letter2_8), 'letter2_8')
            if instrument_read(letter1_8, 'letter1_8') == instrument_read(
                letter2_8, 'letter2_8'):
                print(106, 198)
                diagonal_score_8 = instrument_read_sub(instrument_read_sub(
                    instrument_read(scoring_matrix_8, 'scoring_matrix_8'),
                    'scoring_matrix_8', instrument_read(i_8, 'i_8') - 1,
                    None, None, False), 'scoring_matrix_8[i_8 - 1]', 
                    instrument_read(j_8, 'j_8') - 1, None, None, False
                    ) + instrument_read(match_score_8, 'match_score_8')
                write_instrument_read(diagonal_score_8, 'diagonal_score_8')
                if type(diagonal_score_8) == np.ndarray:
                    print('malloc', sys.getsizeof(diagonal_score_8),
                        'diagonal_score_8', diagonal_score_8.shape)
                elif type(diagonal_score_8) == list:
                    dims = []
                    tmp = diagonal_score_8
                    while type(tmp) == list:
                        dims.append(len(tmp))
                        if len(tmp) > 0:
                            tmp = tmp[0]
                        else:
                            tmp = None
                    print('malloc', sys.getsizeof(diagonal_score_8),
                        'diagonal_score_8', dims)
                elif type(diagonal_score_8) == tuple:
                    print('malloc', sys.getsizeof(diagonal_score_8),
                        'diagonal_score_8', [len(diagonal_score_8)])
                else:
                    print('malloc', sys.getsizeof(diagonal_score_8),
                        'diagonal_score_8')
            else:
                print(108, 201)
                diagonal_score_8 = instrument_read_sub(instrument_read_sub(
                    instrument_read(scoring_matrix_8, 'scoring_matrix_8'),
                    'scoring_matrix_8', instrument_read(i_8, 'i_8') - 1,
                    None, None, False), 'scoring_matrix_8[i_8 - 1]', 
                    instrument_read(j_8, 'j_8') - 1, None, None, False
                    ) + instrument_read(mismatch_score_8, 'mismatch_score_8')
                write_instrument_read(diagonal_score_8, 'diagonal_score_8')
                if type(diagonal_score_8) == np.ndarray:
                    print('malloc', sys.getsizeof(diagonal_score_8),
                        'diagonal_score_8', diagonal_score_8.shape)
                elif type(diagonal_score_8) == list:
                    dims = []
                    tmp = diagonal_score_8
                    while type(tmp) == list:
                        dims.append(len(tmp))
                        if len(tmp) > 0:
                            tmp = tmp[0]
                        else:
                            tmp = None
                    print('malloc', sys.getsizeof(diagonal_score_8),
                        'diagonal_score_8', dims)
                elif type(diagonal_score_8) == tuple:
                    print('malloc', sys.getsizeof(diagonal_score_8),
                        'diagonal_score_8', [len(diagonal_score_8)])
                else:
                    print('malloc', sys.getsizeof(diagonal_score_8),
                        'diagonal_score_8')
            print(107, 203)
            up_score_8 = instrument_read_sub(instrument_read_sub(
                instrument_read(scoring_matrix_8, 'scoring_matrix_8'),
                'scoring_matrix_8', instrument_read(i_8, 'i_8') - 1, None,
                None, False), 'scoring_matrix_8[i_8 - 1]', instrument_read(
                j_8, 'j_8'), None, None, False) + instrument_read(gap_score_8,
                'gap_score_8')
            write_instrument_read(up_score_8, 'up_score_8')
            if type(up_score_8) == np.ndarray:
                print('malloc', sys.getsizeof(up_score_8), 'up_score_8',
                    up_score_8.shape)
            elif type(up_score_8) == list:
                dims = []
                tmp = up_score_8
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(up_score_8), 'up_score_8', dims)
            elif type(up_score_8) == tuple:
                print('malloc', sys.getsizeof(up_score_8), 'up_score_8', [
                    len(up_score_8)])
            else:
                print('malloc', sys.getsizeof(up_score_8), 'up_score_8')
            print(107, 204)
            left_score_8 = instrument_read_sub(instrument_read_sub(
                instrument_read(scoring_matrix_8, 'scoring_matrix_8'),
                'scoring_matrix_8', instrument_read(i_8, 'i_8'), None, None,
                False), 'scoring_matrix_8[i_8]', instrument_read(j_8, 'j_8'
                ) - 1, None, None, False) + instrument_read(gap_score_8,
                'gap_score_8')
            write_instrument_read(left_score_8, 'left_score_8')
            if type(left_score_8) == np.ndarray:
                print('malloc', sys.getsizeof(left_score_8), 'left_score_8',
                    left_score_8.shape)
            elif type(left_score_8) == list:
                dims = []
                tmp = left_score_8
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(left_score_8), 'left_score_8',
                    dims)
            elif type(left_score_8) == tuple:
                print('malloc', sys.getsizeof(left_score_8), 'left_score_8',
                    [len(left_score_8)])
            else:
                print('malloc', sys.getsizeof(left_score_8), 'left_score_8')
            if instrument_read(diagonal_score_8, 'diagonal_score_8'
                ) >= instrument_read(up_score_8, 'up_score_8'):
                if instrument_read(diagonal_score_8, 'diagonal_score_8'
                    ) >= instrument_read(left_score_8, 'left_score_8'):
                    print(115, 207)
                    scoring_matrix_8[instrument_read(instrument_read(i_8,
                        'i_8'), 'i_8')][instrument_read(instrument_read(j_8,
                        'j_8'), 'j_8')] = instrument_read(diagonal_score_8,
                        'diagonal_score_8')
                    write_instrument_read_sub(scoring_matrix_8[
                        instrument_read(instrument_read(i_8, 'i_8'), 'i_8')
                        ], "scoring_matrix_8[instrument_read(i_8, 'i_8')]",
                        instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                        None, None, False)
                    print(115, 208)
                    traceback_matrix_8[instrument_read(instrument_read(i_8,
                        'i_8'), 'i_8')][instrument_read(instrument_read(j_8,
                        'j_8'), 'j_8')] = 'D'
                    write_instrument_read_sub(traceback_matrix_8[
                        instrument_read(instrument_read(i_8, 'i_8'), 'i_8')
                        ],
                        "traceback_matrix_8[instrument_read(i_8, 'i_8')]",
                        instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                        None, None, False)
                else:
                    print(117, 210)
                    scoring_matrix_8[instrument_read(instrument_read(i_8,
                        'i_8'), 'i_8')][instrument_read(instrument_read(j_8,
                        'j_8'), 'j_8')] = instrument_read(left_score_8,
                        'left_score_8')
                    write_instrument_read_sub(scoring_matrix_8[
                        instrument_read(instrument_read(i_8, 'i_8'), 'i_8')
                        ], "scoring_matrix_8[instrument_read(i_8, 'i_8')]",
                        instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                        None, None, False)
                    print(117, 211)
                    traceback_matrix_8[instrument_read(instrument_read(i_8,
                        'i_8'), 'i_8')][instrument_read(instrument_read(j_8,
                        'j_8'), 'j_8')] = 'L'
                    write_instrument_read_sub(traceback_matrix_8[
                        instrument_read(instrument_read(i_8, 'i_8'), 'i_8')
                        ],
                        "traceback_matrix_8[instrument_read(i_8, 'i_8')]",
                        instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                        None, None, False)
            elif instrument_read(up_score_8, 'up_score_8') >= instrument_read(
                left_score_8, 'left_score_8'):
                print(112, 213)
                scoring_matrix_8[instrument_read(instrument_read(i_8, 'i_8'
                    ), 'i_8')][instrument_read(instrument_read(j_8, 'j_8'),
                    'j_8')] = instrument_read(up_score_8, 'up_score_8')
                write_instrument_read_sub(scoring_matrix_8[instrument_read(
                    instrument_read(i_8, 'i_8'), 'i_8')],
                    "scoring_matrix_8[instrument_read(i_8, 'i_8')]",
                    instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                    None, None, False)
                print(112, 214)
                traceback_matrix_8[instrument_read(instrument_read(i_8,
                    'i_8'), 'i_8')][instrument_read(instrument_read(j_8,
                    'j_8'), 'j_8')] = 'U'
                write_instrument_read_sub(traceback_matrix_8[
                    instrument_read(instrument_read(i_8, 'i_8'), 'i_8')],
                    "traceback_matrix_8[instrument_read(i_8, 'i_8')]",
                    instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                    None, None, False)
            else:
                print(114, 216)
                scoring_matrix_8[instrument_read(instrument_read(i_8, 'i_8'
                    ), 'i_8')][instrument_read(instrument_read(j_8, 'j_8'),
                    'j_8')] = instrument_read(left_score_8, 'left_score_8')
                write_instrument_read_sub(scoring_matrix_8[instrument_read(
                    instrument_read(i_8, 'i_8'), 'i_8')],
                    "scoring_matrix_8[instrument_read(i_8, 'i_8')]",
                    instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                    None, None, False)
                print(114, 217)
                traceback_matrix_8[instrument_read(instrument_read(i_8,
                    'i_8'), 'i_8')][instrument_read(instrument_read(j_8,
                    'j_8'), 'j_8')] = 'L'
                write_instrument_read_sub(traceback_matrix_8[
                    instrument_read(instrument_read(i_8, 'i_8'), 'i_8')],
                    "traceback_matrix_8[instrument_read(i_8, 'i_8')]",
                    instrument_read(instrument_read(j_8, 'j_8'), 'j_8'),
                    None, None, False)
            if instrument_read_sub(instrument_read_sub(instrument_read(
                scoring_matrix_8, 'scoring_matrix_8'), 'scoring_matrix_8',
                instrument_read(i_8, 'i_8'), None, None, False),
                'scoring_matrix_8[i_8]', instrument_read(j_8, 'j_8'), None,
                None, False) >= instrument_read(max_score_8, 'max_score_8'):
                print(118, 219)
                max_score_8 = instrument_read_sub(instrument_read_sub(
                    instrument_read(scoring_matrix_8, 'scoring_matrix_8'),
                    'scoring_matrix_8', instrument_read(i_8, 'i_8'), None,
                    None, False), 'scoring_matrix_8[i_8]', instrument_read(
                    j_8, 'j_8'), None, None, False)
                write_instrument_read(max_score_8, 'max_score_8')
                if type(max_score_8) == np.ndarray:
                    print('malloc', sys.getsizeof(max_score_8),
                        'max_score_8', max_score_8.shape)
                elif type(max_score_8) == list:
                    dims = []
                    tmp = max_score_8
                    while type(tmp) == list:
                        dims.append(len(tmp))
                        if len(tmp) > 0:
                            tmp = tmp[0]
                        else:
                            tmp = None
                    print('malloc', sys.getsizeof(max_score_8),
                        'max_score_8', dims)
                elif type(max_score_8) == tuple:
                    print('malloc', sys.getsizeof(max_score_8),
                        'max_score_8', [len(max_score_8)])
                else:
                    print('malloc', sys.getsizeof(max_score_8), 'max_score_8')
                print(118, 220)
                max_pos_8 = instrument_read(i_8, 'i_8'), instrument_read(j_8,
                    'j_8')
                write_instrument_read(max_pos_8, 'max_pos_8')
                if type(max_pos_8) == np.ndarray:
                    print('malloc', sys.getsizeof(max_pos_8), 'max_pos_8',
                        max_pos_8.shape)
                elif type(max_pos_8) == list:
                    dims = []
                    tmp = max_pos_8
                    while type(tmp) == list:
                        dims.append(len(tmp))
                        if len(tmp) > 0:
                            tmp = tmp[0]
                        else:
                            tmp = None
                    print('malloc', sys.getsizeof(max_pos_8), 'max_pos_8', dims
                        )
                elif type(max_pos_8) == tuple:
                    print('malloc', sys.getsizeof(max_pos_8), 'max_pos_8',
                        [len(max_pos_8)])
                else:
                    print('malloc', sys.getsizeof(max_pos_8), 'max_pos_8')
    print(103, 221)
    i_8 = instrument_read(max_pos_8, 'max_pos_8')
    write_instrument_read(i_8, 'i_8')
    if type(i_8) == np.ndarray:
        print('malloc', sys.getsizeof(i_8), 'i_8', i_8.shape)
    elif type(i_8) == list:
        dims = []
        tmp = i_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(i_8), 'i_8', dims)
    elif type(i_8) == tuple:
        print('malloc', sys.getsizeof(i_8), 'i_8', [len(i_8)])
    else:
        print('malloc', sys.getsizeof(i_8), 'i_8')
    print(103, 222)
    j_8 = instrument_read(max_pos_8, 'max_pos_8')
    write_instrument_read(j_8, 'j_8')
    if type(j_8) == np.ndarray:
        print('malloc', sys.getsizeof(j_8), 'j_8', j_8.shape)
    elif type(j_8) == list:
        dims = []
        tmp = j_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(j_8), 'j_8', dims)
    elif type(j_8) == tuple:
        print('malloc', sys.getsizeof(j_8), 'j_8', [len(j_8)])
    else:
        print('malloc', sys.getsizeof(j_8), 'j_8')
    print(103, 223)
    aln1_8 = ''
    write_instrument_read(aln1_8, 'aln1_8')
    if type(aln1_8) == np.ndarray:
        print('malloc', sys.getsizeof(aln1_8), 'aln1_8', aln1_8.shape)
    elif type(aln1_8) == list:
        dims = []
        tmp = aln1_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(aln1_8), 'aln1_8', dims)
    elif type(aln1_8) == tuple:
        print('malloc', sys.getsizeof(aln1_8), 'aln1_8', [len(aln1_8)])
    else:
        print('malloc', sys.getsizeof(aln1_8), 'aln1_8')
    print(103, 224)
    aln2_8 = ''
    write_instrument_read(aln2_8, 'aln2_8')
    if type(aln2_8) == np.ndarray:
        print('malloc', sys.getsizeof(aln2_8), 'aln2_8', aln2_8.shape)
    elif type(aln2_8) == list:
        dims = []
        tmp = aln2_8
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(aln2_8), 'aln2_8', dims)
    elif type(aln2_8) == tuple:
        print('malloc', sys.getsizeof(aln2_8), 'aln2_8', [len(aln2_8)])
    else:
        print('malloc', sys.getsizeof(aln2_8), 'aln2_8')
    while instrument_read_sub(instrument_read_sub(instrument_read(
        traceback_matrix_8, 'traceback_matrix_8'), 'traceback_matrix_8',
        instrument_read(i_8, 'i_8'), None, None, False),
        'traceback_matrix_8[i_8]', instrument_read(j_8, 'j_8'), None, None,
        False) != None:
        if instrument_read_sub(instrument_read_sub(instrument_read(
            traceback_matrix_8, 'traceback_matrix_8'), 'traceback_matrix_8',
            instrument_read(i_8, 'i_8'), None, None, False),
            'traceback_matrix_8[i_8]', instrument_read(j_8, 'j_8'), None,
            None, False) == 'D':
            print(123, 227)
            aln1_8 += instrument_read_sub(instrument_read(seq1_8, 'seq1_8'),
                'seq1_8', instrument_read(i_8, 'i_8') - 1, None, None, False)
            write_instrument_read(aln1_8, 'aln1_8')
            print(123, 228)
            aln2_8 += instrument_read_sub(instrument_read(seq2_8, 'seq2_8'),
                'seq2_8', instrument_read(j_8, 'j_8') - 1, None, None, False)
            write_instrument_read(aln2_8, 'aln2_8')
            print(123, 229)
            i_8 -= 1
            write_instrument_read(i_8, 'i_8')
            print(123, 230)
            j_8 -= 1
            write_instrument_read(j_8, 'j_8')
        elif instrument_read_sub(instrument_read_sub(instrument_read(
            traceback_matrix_8, 'traceback_matrix_8'), 'traceback_matrix_8',
            instrument_read(i_8, 'i_8'), None, None, False),
            'traceback_matrix_8[i_8]', instrument_read(j_8, 'j_8'), None,
            None, False) == 'L':
            print(126, 232)
            aln1_8 += '-'
            write_instrument_read(aln1_8, 'aln1_8')
            print(126, 233)
            aln2_8 += instrument_read_sub(instrument_read(seq2_8, 'seq2_8'),
                'seq2_8', instrument_read(j_8, 'j_8') - 1, None, None, False)
            write_instrument_read(aln2_8, 'aln2_8')
            print(126, 234)
            j_8 -= 1
            write_instrument_read(j_8, 'j_8')
        elif instrument_read_sub(instrument_read_sub(instrument_read(
            traceback_matrix_8, 'traceback_matrix_8'), 'traceback_matrix_8',
            instrument_read(i_8, 'i_8'), None, None, False),
            'traceback_matrix_8[i_8]', instrument_read(j_8, 'j_8'), None,
            None, False) == 'U':
            print(129, 236)
            aln1_8 += instrument_read_sub(instrument_read(seq1_8, 'seq1_8'),
                'seq1_8', instrument_read(i_8, 'i_8') - 1, None, None, False)
            write_instrument_read(aln1_8, 'aln1_8')
            print(129, 237)
            aln2_8 += '-'
            write_instrument_read(aln2_8, 'aln2_8')
            print(129, 238)
            i_8 -= 1
            write_instrument_read(i_8, 'i_8')
    print('exit scope 8')
    return instrument_read_sub(instrument_read(aln1_8, 'aln1_8'), 'aln1_8',
        None, None, None, True), instrument_read_sub(instrument_read(aln2_8,
        'aln2_8'), 'aln2_8', None, None, None, True)
    print('exit scope 8')


def main():
    print('enter scope 9')
    print(1, 242)
    print(134, 243)
    fasta_file_9 = instrument_read(path_0, 'path_0'
        ) + '/benchmarks/supplemental_files/fasta_example.fasta'
    write_instrument_read(fasta_file_9, 'fasta_file_9')
    if type(fasta_file_9) == np.ndarray:
        print('malloc', sys.getsizeof(fasta_file_9), 'fasta_file_9',
            fasta_file_9.shape)
    elif type(fasta_file_9) == list:
        dims = []
        tmp = fasta_file_9
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fasta_file_9), 'fasta_file_9', dims)
    elif type(fasta_file_9) == tuple:
        print('malloc', sys.getsizeof(fasta_file_9), 'fasta_file_9', [len(
            fasta_file_9)])
    else:
        print('malloc', sys.getsizeof(fasta_file_9), 'fasta_file_9')
    print(134, 245)
    fastq_file_9 = instrument_read(path_0, 'path_0'
        ) + '/benchmarks/supplemental_files/fastq_large.fastq'
    write_instrument_read(fastq_file_9, 'fastq_file_9')
    if type(fastq_file_9) == np.ndarray:
        print('malloc', sys.getsizeof(fastq_file_9), 'fastq_file_9',
            fastq_file_9.shape)
    elif type(fastq_file_9) == list:
        dims = []
        tmp = fastq_file_9
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(fastq_file_9), 'fastq_file_9', dims)
    elif type(fastq_file_9) == tuple:
        print('malloc', sys.getsizeof(fastq_file_9), 'fastq_file_9', [len(
            fastq_file_9)])
    else:
        print('malloc', sys.getsizeof(fastq_file_9), 'fastq_file_9')
    print(134, 246)
    output_file_9 = instrument_read(path_0, 'path_0'
        ) + '/benchmarks/supplemental_files/output_file.txt'
    write_instrument_read(output_file_9, 'output_file_9')
    if type(output_file_9) == np.ndarray:
        print('malloc', sys.getsizeof(output_file_9), 'output_file_9',
            output_file_9.shape)
    elif type(output_file_9) == list:
        dims = []
        tmp = output_file_9
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(output_file_9), 'output_file_9', dims)
    elif type(output_file_9) == tuple:
        print('malloc', sys.getsizeof(output_file_9), 'output_file_9', [len
            (output_file_9)])
    else:
        print('malloc', sys.getsizeof(output_file_9), 'output_file_9')
    darwin_wga_workflow(instrument_read(fasta_file_9, 'fasta_file_9'),
        instrument_read(fastq_file_9, 'fastq_file_9'), instrument_read(
        output_file_9, 'output_file_9'))
    print('exit scope 9')


if instrument_read(__name__, '__name__') == '__main__':
    main()
