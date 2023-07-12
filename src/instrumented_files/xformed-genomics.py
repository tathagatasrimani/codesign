import sys
from instrument_lib import *
import sys
from instrument_lib import *


def read_fasta(fasta_file):
    print('enter scope 1')
    print(1, 3)
    fasta_file__1 = instrument_read(fasta_file, 'fasta_file')
    write_instrument_read(fasta_file__1, 'fasta_file__1')
    print('malloc', sys.getsizeof(fasta_file__1), 'fasta_file__1')
    """
    Reads a fasta file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(3, 8)
    fasta_dict__1 = {}
    write_instrument_read(fasta_dict__1, 'fasta_dict__1')
    print('malloc', sys.getsizeof(fasta_dict__1), 'fasta_dict__1')
    with open(instrument_read(fasta_file__1, 'fasta_file__1'), 'r') as f__1:
        for line__1 in instrument_read(f__1, 'f__1'):
            print(5, 11)
            line__1 = instrument_read(line__1, 'line__1').strip()
            write_instrument_read(line__1, 'line__1')
            print('malloc', sys.getsizeof(line__1), 'line__1')
            if not instrument_read(line__1, 'line__1'):
                continue
            if instrument_read(line__1, 'line__1').startswith('>'):
                print(9, 15)
                active_sequence_name__1 = instrument_read_sub(instrument_read
                    (line__1, 'line__1'), 'line__1', None, 1, None, True)
                write_instrument_read(active_sequence_name__1,
                    'active_sequence_name__1')
                print('malloc', sys.getsizeof(active_sequence_name__1),
                    'active_sequence_name__1')
                if instrument_read(active_sequence_name__1,
                    'active_sequence_name__1') not in instrument_read(
                    fasta_dict__1, 'fasta_dict__1'):
                    print(11, 17)
                    fasta_dict__1[instrument_read(instrument_read(
                        active_sequence_name__1, 'active_sequence_name__1'),
                        'active_sequence_name__1')] = []
                    write_instrument_read_sub(fasta_dict__1,
                        'fasta_dict__1', instrument_read(instrument_read(
                        active_sequence_name__1, 'active_sequence_name__1'),
                        'active_sequence_name__1'), None, None, False)
                continue
            print(10, 19)
            sequence__1 = instrument_read(line__1, 'line__1')
            write_instrument_read(sequence__1, 'sequence__1')
            print('malloc', sys.getsizeof(sequence__1), 'sequence__1')
            instrument_read_sub(instrument_read(fasta_dict__1,
                'fasta_dict__1'), 'fasta_dict__1', instrument_read(
                active_sequence_name__1, 'active_sequence_name__1'), None,
                None, False).append(instrument_read(sequence__1, 'sequence__1')
                )
    print('exit scope 1')
    return instrument_read(fasta_dict__1, 'fasta_dict__1')
    print('exit scope 1')


def read_fastq(fastq_file):
    print('enter scope 2')
    print(1, 24)
    fastq_file__2 = instrument_read(fastq_file, 'fastq_file')
    write_instrument_read(fastq_file__2, 'fastq_file__2')
    print('malloc', sys.getsizeof(fastq_file__2), 'fastq_file__2')
    """
    Reads a fastq file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(16, 29)
    fastq_dict__2 = {}
    write_instrument_read(fastq_dict__2, 'fastq_dict__2')
    print('malloc', sys.getsizeof(fastq_dict__2), 'fastq_dict__2')
    with open(instrument_read(fastq_file__2, 'fastq_file__2'), 'r') as f__2:
        for line__2 in instrument_read(f__2, 'f__2'):
            print(18, 32)
            line__2 = instrument_read(line__2, 'line__2').strip()
            write_instrument_read(line__2, 'line__2')
            print('malloc', sys.getsizeof(line__2), 'line__2')
            if not instrument_read(line__2, 'line__2'):
                continue
            if instrument_read(line__2, 'line__2').startswith('@'):
                print(22, 36)
                active_sequence_name__2 = instrument_read_sub(instrument_read
                    (line__2, 'line__2'), 'line__2', None, 1, None, True)
                write_instrument_read(active_sequence_name__2,
                    'active_sequence_name__2')
                print('malloc', sys.getsizeof(active_sequence_name__2),
                    'active_sequence_name__2')
                if instrument_read(active_sequence_name__2,
                    'active_sequence_name__2') not in instrument_read(
                    fastq_dict__2, 'fastq_dict__2'):
                    print(24, 38)
                    fastq_dict__2[instrument_read(instrument_read(
                        active_sequence_name__2, 'active_sequence_name__2'),
                        'active_sequence_name__2')] = []
                    write_instrument_read_sub(fastq_dict__2,
                        'fastq_dict__2', instrument_read(instrument_read(
                        active_sequence_name__2, 'active_sequence_name__2'),
                        'active_sequence_name__2'), None, None, False)
                continue
            print(23, 40)
            sequence__2 = instrument_read(line__2, 'line__2')
            write_instrument_read(sequence__2, 'sequence__2')
            print('malloc', sys.getsizeof(sequence__2), 'sequence__2')
            instrument_read_sub(instrument_read(fastq_dict__2,
                'fastq_dict__2'), 'fastq_dict__2', instrument_read(
                active_sequence_name__2, 'active_sequence_name__2'), None,
                None, False).append(instrument_read(sequence__2, 'sequence__2')
                )
    print('exit scope 2')
    return instrument_read(fastq_dict__2, 'fastq_dict__2')
    print('exit scope 2')


def read_fasta_with_quality(fasta_file, quality_file):
    print('enter scope 3')
    print(1, 45)
    fasta_file__3 = instrument_read(fasta_file, 'fasta_file')
    write_instrument_read(fasta_file__3, 'fasta_file__3')
    print('malloc', sys.getsizeof(fasta_file__3), 'fasta_file__3')
    quality_file__3 = instrument_read(quality_file, 'quality_file')
    write_instrument_read(quality_file__3, 'quality_file__3')
    print('malloc', sys.getsizeof(quality_file__3), 'quality_file__3')
    """
    Reads a fasta file and a quality file and returns a dictionary with sequence
    number as keys and sequence code as values
    """
    print(29, 50)
    fasta_dict__3 = {}
    write_instrument_read(fasta_dict__3, 'fasta_dict__3')
    print('malloc', sys.getsizeof(fasta_dict__3), 'fasta_dict__3')
    with open(instrument_read(fasta_file__3, 'fasta_file__3'), 'r') as f__3:
        for line__3 in instrument_read(f__3, 'f__3'):
            print(31, 53)
            line__3 = instrument_read(line__3, 'line__3').strip()
            write_instrument_read(line__3, 'line__3')
            print('malloc', sys.getsizeof(line__3), 'line__3')
            if not instrument_read(line__3, 'line__3'):
                continue
            if instrument_read(line__3, 'line__3').startswith('>'):
                print(35, 57)
                active_sequence_name__3 = instrument_read_sub(instrument_read
                    (line__3, 'line__3'), 'line__3', None, 1, None, True)
                write_instrument_read(active_sequence_name__3,
                    'active_sequence_name__3')
                print('malloc', sys.getsizeof(active_sequence_name__3),
                    'active_sequence_name__3')
                if instrument_read(active_sequence_name__3,
                    'active_sequence_name__3') not in instrument_read(
                    fasta_dict__3, 'fasta_dict__3'):
                    print(37, 59)
                    fasta_dict__3[instrument_read(instrument_read(
                        active_sequence_name__3, 'active_sequence_name__3'),
                        'active_sequence_name__3')] = []
                    write_instrument_read_sub(fasta_dict__3,
                        'fasta_dict__3', instrument_read(instrument_read(
                        active_sequence_name__3, 'active_sequence_name__3'),
                        'active_sequence_name__3'), None, None, False)
                continue
            print(36, 61)
            sequence__3 = instrument_read(line__3, 'line__3')
            write_instrument_read(sequence__3, 'sequence__3')
            print('malloc', sys.getsizeof(sequence__3), 'sequence__3')
            instrument_read_sub(instrument_read(fasta_dict__3,
                'fasta_dict__3'), 'fasta_dict__3', instrument_read(
                active_sequence_name__3, 'active_sequence_name__3'), None,
                None, False).append(instrument_read(sequence__3, 'sequence__3')
                )
    print(32, 63)
    quality_dict__3 = {}
    write_instrument_read(quality_dict__3, 'quality_dict__3')
    print('malloc', sys.getsizeof(quality_dict__3), 'quality_dict__3')
    with open(instrument_read(quality_file__3, 'quality_file__3'), 'r'
        ) as f__3:
        for line__3 in instrument_read(f__3, 'f__3'):
            print(40, 66)
            line__3 = instrument_read(line__3, 'line__3').strip()
            write_instrument_read(line__3, 'line__3')
            print('malloc', sys.getsizeof(line__3), 'line__3')
            if not instrument_read(line__3, 'line__3'):
                continue
            if instrument_read(line__3, 'line__3').startswith('>'):
                print(44, 70)
                active_sequence_name__3 = instrument_read_sub(instrument_read
                    (line__3, 'line__3'), 'line__3', None, 1, None, True)
                write_instrument_read(active_sequence_name__3,
                    'active_sequence_name__3')
                print('malloc', sys.getsizeof(active_sequence_name__3),
                    'active_sequence_name__3')
                if instrument_read(active_sequence_name__3,
                    'active_sequence_name__3') not in instrument_read(
                    quality_dict__3, 'quality_dict__3'):
                    print(46, 72)
                    quality_dict__3[instrument_read(instrument_read(
                        active_sequence_name__3, 'active_sequence_name__3'),
                        'active_sequence_name__3')] = []
                    write_instrument_read_sub(quality_dict__3,
                        'quality_dict__3', instrument_read(instrument_read(
                        active_sequence_name__3, 'active_sequence_name__3'),
                        'active_sequence_name__3'), None, None, False)
                continue
            print(45, 74)
            quality__3 = instrument_read(line__3, 'line__3')
            write_instrument_read(quality__3, 'quality__3')
            print('malloc', sys.getsizeof(quality__3), 'quality__3')
            instrument_read_sub(instrument_read(quality_dict__3,
                'quality_dict__3'), 'quality_dict__3', instrument_read(
                active_sequence_name__3, 'active_sequence_name__3'), None,
                None, False).append(instrument_read(quality__3, 'quality__3'))
    print('exit scope 3')
    return instrument_read(fasta_dict__3, 'fasta_dict__3'), instrument_read(
        quality_dict__3, 'quality_dict__3')
    print('exit scope 3')


def darwin_wga_workflow(fasta_file, fastq_file, output_file, min_length=0,
    max_length=0, min_quality=0, max_quality=0, min_length_fraction=0,
    max_length_fraction=0):
    print('enter scope 4')
    print(1, 79)
    fasta_file__4 = instrument_read(fasta_file, 'fasta_file')
    write_instrument_read(fasta_file__4, 'fasta_file__4')
    print('malloc', sys.getsizeof(fasta_file__4), 'fasta_file__4')
    fastq_file__4 = instrument_read(fastq_file, 'fastq_file')
    write_instrument_read(fastq_file__4, 'fastq_file__4')
    print('malloc', sys.getsizeof(fastq_file__4), 'fastq_file__4')
    output_file__4 = instrument_read(output_file, 'output_file')
    write_instrument_read(output_file__4, 'output_file__4')
    print('malloc', sys.getsizeof(output_file__4), 'output_file__4')
    min_length__4 = instrument_read(min_length, 'min_length')
    write_instrument_read(min_length__4, 'min_length__4')
    print('malloc', sys.getsizeof(min_length__4), 'min_length__4')
    max_length__4 = instrument_read(max_length, 'max_length')
    write_instrument_read(max_length__4, 'max_length__4')
    print('malloc', sys.getsizeof(max_length__4), 'max_length__4')
    min_quality__4 = instrument_read(min_quality, 'min_quality')
    write_instrument_read(min_quality__4, 'min_quality__4')
    print('malloc', sys.getsizeof(min_quality__4), 'min_quality__4')
    max_quality__4 = instrument_read(max_quality, 'max_quality')
    write_instrument_read(max_quality__4, 'max_quality__4')
    print('malloc', sys.getsizeof(max_quality__4), 'max_quality__4')
    min_length_fraction__4 = instrument_read(min_length_fraction,
        'min_length_fraction')
    write_instrument_read(min_length_fraction__4, 'min_length_fraction__4')
    print('malloc', sys.getsizeof(min_length_fraction__4),
        'min_length_fraction__4')
    max_length_fraction__4 = instrument_read(max_length_fraction,
        'max_length_fraction')
    write_instrument_read(max_length_fraction__4, 'max_length_fraction__4')
    print('malloc', sys.getsizeof(max_length_fraction__4),
        'max_length_fraction__4')
    """
    DARWIN-Whole Genome Alignment workflow
    """
    print(51, 87)
    fasta_dict__4 = read_fasta(instrument_read(fasta_file__4, 'fasta_file__4'))
    write_instrument_read(fasta_dict__4, 'fasta_dict__4')
    print('malloc', sys.getsizeof(fasta_dict__4), 'fasta_dict__4')
    print(51, 89)
    fastq_dict__4 = read_fastq(instrument_read(fastq_file__4, 'fastq_file__4'))
    write_instrument_read(fastq_dict__4, 'fastq_dict__4')
    print('malloc', sys.getsizeof(fastq_dict__4), 'fastq_dict__4')
    print(51, 91)
    filtered_fasta_dict__4 = filter_fasta(instrument_read(fasta_dict__4,
        'fasta_dict__4'), instrument_read(min_length__4, 'min_length__4'),
        instrument_read(max_length__4, 'max_length__4'), instrument_read(
        min_length_fraction__4, 'min_length_fraction__4'), instrument_read(
        max_length_fraction__4, 'max_length_fraction__4'))
    write_instrument_read(filtered_fasta_dict__4, 'filtered_fasta_dict__4')
    print('malloc', sys.getsizeof(filtered_fasta_dict__4),
        'filtered_fasta_dict__4')
    print(51, 94)
    filtered_fastq_dict__4 = filter_fastq(instrument_read(fastq_dict__4,
        'fastq_dict__4'), instrument_read(min_quality__4, 'min_quality__4'),
        instrument_read(max_quality__4, 'max_quality__4'))
    write_instrument_read(filtered_fastq_dict__4, 'filtered_fastq_dict__4')
    print('malloc', sys.getsizeof(filtered_fastq_dict__4),
        'filtered_fastq_dict__4')
    write_output(instrument_read(filtered_fasta_dict__4,
        'filtered_fasta_dict__4'), instrument_read(filtered_fastq_dict__4,
        'filtered_fastq_dict__4'), instrument_read(output_file__4,
        'output_file__4'))
    print('exit scope 4')


def filter_fasta(fasta_dict, min_length, max_length, min_length_fraction,
    max_length_fraction):
    print('enter scope 5')
    print(1, 100)
    fasta_dict__5 = instrument_read(fasta_dict, 'fasta_dict')
    write_instrument_read(fasta_dict__5, 'fasta_dict__5')
    print('malloc', sys.getsizeof(fasta_dict__5), 'fasta_dict__5')
    min_length__5 = instrument_read(min_length, 'min_length')
    write_instrument_read(min_length__5, 'min_length__5')
    print('malloc', sys.getsizeof(min_length__5), 'min_length__5')
    max_length__5 = instrument_read(max_length, 'max_length')
    write_instrument_read(max_length__5, 'max_length__5')
    print('malloc', sys.getsizeof(max_length__5), 'max_length__5')
    min_length_fraction__5 = instrument_read(min_length_fraction,
        'min_length_fraction')
    write_instrument_read(min_length_fraction__5, 'min_length_fraction__5')
    print('malloc', sys.getsizeof(min_length_fraction__5),
        'min_length_fraction__5')
    max_length_fraction__5 = instrument_read(max_length_fraction,
        'max_length_fraction')
    write_instrument_read(max_length_fraction__5, 'max_length_fraction__5')
    print('malloc', sys.getsizeof(max_length_fraction__5),
        'max_length_fraction__5')
    """
    Filters a fasta dictionary by length
    """
    print(54, 104)
    filtered_fasta_dict__5 = {}
    write_instrument_read(filtered_fasta_dict__5, 'filtered_fasta_dict__5')
    print('malloc', sys.getsizeof(filtered_fasta_dict__5),
        'filtered_fasta_dict__5')
    for key__5 in instrument_read(fasta_dict__5, 'fasta_dict__5'):
        print(56, 106)
        sequence__5 = instrument_read_sub(instrument_read_sub(
            instrument_read(fasta_dict__5, 'fasta_dict__5'),
            'fasta_dict__5', instrument_read(key__5, 'key__5'), None, None,
            False), 'fasta_dict__5[key__5]', 0, None, None, False)
        write_instrument_read(sequence__5, 'sequence__5')
        print('malloc', sys.getsizeof(sequence__5), 'sequence__5')
        print(56, 107)
        length__5 = len(instrument_read(sequence__5, 'sequence__5'))
        write_instrument_read(length__5, 'length__5')
        print('malloc', sys.getsizeof(length__5), 'length__5')
        if instrument_read(min_length_fraction__5, 'min_length_fraction__5'
            ) > 0:
            if instrument_read(length__5, 'length__5') < instrument_read(
                min_length_fraction__5, 'min_length_fraction__5') * len(
                instrument_read_sub(instrument_read_sub(instrument_read(
                fasta_dict__5, 'fasta_dict__5'), 'fasta_dict__5',
                instrument_read(key__5, 'key__5'), None, None, False),
                'fasta_dict__5[key__5]', 0, None, None, False)):
                continue
        if instrument_read(max_length_fraction__5, 'max_length_fraction__5'
            ) > 0:
            if instrument_read(length__5, 'length__5') > instrument_read(
                max_length_fraction__5, 'max_length_fraction__5') * len(
                instrument_read_sub(instrument_read_sub(instrument_read(
                fasta_dict__5, 'fasta_dict__5'), 'fasta_dict__5',
                instrument_read(key__5, 'key__5'), None, None, False),
                'fasta_dict__5[key__5]', 0, None, None, False)):
                continue
        if instrument_read(min_length__5, 'min_length__5') > 0:
            if instrument_read(length__5, 'length__5') < instrument_read(
                min_length__5, 'min_length__5'):
                continue
        if instrument_read(max_length__5, 'max_length__5') > 0:
            if instrument_read(length__5, 'length__5') > instrument_read(
                max_length__5, 'max_length__5'):
                continue
        print(71, 120)
        filtered_fasta_dict__5[instrument_read(instrument_read(key__5,
            'key__5'), 'key__5')] = instrument_read_sub(instrument_read(
            fasta_dict__5, 'fasta_dict__5'), 'fasta_dict__5',
            instrument_read(key__5, 'key__5'), None, None, False)
        write_instrument_read_sub(filtered_fasta_dict__5,
            'filtered_fasta_dict__5', instrument_read(instrument_read(
            key__5, 'key__5'), 'key__5'), None, None, False)
    print('exit scope 5')
    return instrument_read(filtered_fasta_dict__5, 'filtered_fasta_dict__5')
    print('exit scope 5')


def filter_fastq(fastq_dict, min_quality, max_quality):
    print('enter scope 6')
    print(1, 124)
    fastq_dict__6 = instrument_read(fastq_dict, 'fastq_dict')
    write_instrument_read(fastq_dict__6, 'fastq_dict__6')
    print('malloc', sys.getsizeof(fastq_dict__6), 'fastq_dict__6')
    min_quality__6 = instrument_read(min_quality, 'min_quality')
    write_instrument_read(min_quality__6, 'min_quality__6')
    print('malloc', sys.getsizeof(min_quality__6), 'min_quality__6')
    max_quality__6 = instrument_read(max_quality, 'max_quality')
    write_instrument_read(max_quality__6, 'max_quality__6')
    print('malloc', sys.getsizeof(max_quality__6), 'max_quality__6')
    """
    Filters a fastq dictionary by quality
    """
    print(77, 128)
    filtered_fastq_dict__6 = {}
    write_instrument_read(filtered_fastq_dict__6, 'filtered_fastq_dict__6')
    print('malloc', sys.getsizeof(filtered_fastq_dict__6),
        'filtered_fastq_dict__6')
    for key__6 in instrument_read(fastq_dict__6, 'fastq_dict__6'):
        print(79, 130)
        quality__6 = instrument_read_sub(instrument_read_sub(
            instrument_read(fastq_dict__6, 'fastq_dict__6'),
            'fastq_dict__6', instrument_read(key__6, 'key__6'), None, None,
            False), 'fastq_dict__6[key__6]', 0, None, None, False)
        write_instrument_read(quality__6, 'quality__6')
        print('malloc', sys.getsizeof(quality__6), 'quality__6')
        if instrument_read(min_quality__6, 'min_quality__6') > 0:
            if instrument_read(quality__6, 'quality__6') < instrument_read(
                min_quality__6, 'min_quality__6'):
                continue
        if instrument_read(max_quality__6, 'max_quality__6') > 0:
            if instrument_read(quality__6, 'quality__6') > instrument_read(
                max_quality__6, 'max_quality__6'):
                continue
        print(86, 137)
        filtered_fastq_dict__6[instrument_read(instrument_read(key__6,
            'key__6'), 'key__6')] = instrument_read_sub(instrument_read(
            fastq_dict__6, 'fastq_dict__6'), 'fastq_dict__6',
            instrument_read(key__6, 'key__6'), None, None, False)
        write_instrument_read_sub(filtered_fastq_dict__6,
            'filtered_fastq_dict__6', instrument_read(instrument_read(
            key__6, 'key__6'), 'key__6'), None, None, False)
    print('exit scope 6')
    return instrument_read(filtered_fastq_dict__6, 'filtered_fastq_dict__6')
    print('exit scope 6')


def write_output(filtered_fasta_dict, filtered_fastq_dict, output_file):
    print('enter scope 7')
    print(1, 141)
    filtered_fasta_dict__7 = instrument_read(filtered_fasta_dict,
        'filtered_fasta_dict')
    write_instrument_read(filtered_fasta_dict__7, 'filtered_fasta_dict__7')
    print('malloc', sys.getsizeof(filtered_fasta_dict__7),
        'filtered_fasta_dict__7')
    filtered_fastq_dict__7 = instrument_read(filtered_fastq_dict,
        'filtered_fastq_dict')
    write_instrument_read(filtered_fastq_dict__7, 'filtered_fastq_dict__7')
    print('malloc', sys.getsizeof(filtered_fastq_dict__7),
        'filtered_fastq_dict__7')
    output_file__7 = instrument_read(output_file, 'output_file')
    write_instrument_read(output_file__7, 'output_file__7')
    print('malloc', sys.getsizeof(output_file__7), 'output_file__7')
    """
    Writes output
    """
    with open(instrument_read(output_file__7, 'output_file__7'), 'w') as f__7:
        for key__7 in instrument_read(filtered_fasta_dict__7,
            'filtered_fasta_dict__7'):
            instrument_read(f__7, 'f__7').write('>' + instrument_read(
                key__7, 'key__7') + '\n')
            instrument_read(f__7, 'f__7').write(instrument_read_sub(
                instrument_read_sub(instrument_read(filtered_fasta_dict__7,
                'filtered_fasta_dict__7'), 'filtered_fasta_dict__7',
                instrument_read(key__7, 'key__7'), None, None, False),
                'filtered_fasta_dict__7[key__7]', 0, None, None, False) + '\n')
        for key__7 in instrument_read(filtered_fastq_dict__7,
            'filtered_fastq_dict__7'):
            instrument_read(f__7, 'f__7').write('@' + instrument_read(
                key__7, 'key__7') + '\n')
            instrument_read(f__7, 'f__7').write(instrument_read_sub(
                instrument_read_sub(instrument_read(filtered_fastq_dict__7,
                'filtered_fastq_dict__7'), 'filtered_fastq_dict__7',
                instrument_read(key__7, 'key__7'), None, None, False),
                'filtered_fastq_dict__7[key__7]', 0, None, None, False) + '\n')
            instrument_read(f__7, 'f__7').write('+\n')
            instrument_read(f__7, 'f__7').write(instrument_read_sub(
                instrument_read_sub(instrument_read(filtered_fastq_dict__7,
                'filtered_fastq_dict__7'), 'filtered_fastq_dict__7',
                instrument_read(key__7, 'key__7'), None, None, False),
                'filtered_fastq_dict__7[key__7]', 1, None, None, False) + '\n')
    print('exit scope 7')


def smith_waterman(seq1, seq2, match_score=3, mismatch_score=-3, gap_score=-2):
    print('enter scope 8')
    print(1, 159)
    seq1__8 = instrument_read(seq1, 'seq1')
    write_instrument_read(seq1__8, 'seq1__8')
    print('malloc', sys.getsizeof(seq1__8), 'seq1__8')
    seq2__8 = instrument_read(seq2, 'seq2')
    write_instrument_read(seq2__8, 'seq2__8')
    print('malloc', sys.getsizeof(seq2__8), 'seq2__8')
    match_score__8 = instrument_read(match_score, 'match_score')
    write_instrument_read(match_score__8, 'match_score__8')
    print('malloc', sys.getsizeof(match_score__8), 'match_score__8')
    mismatch_score__8 = instrument_read(mismatch_score, 'mismatch_score')
    write_instrument_read(mismatch_score__8, 'mismatch_score__8')
    print('malloc', sys.getsizeof(mismatch_score__8), 'mismatch_score__8')
    gap_score__8 = instrument_read(gap_score, 'gap_score')
    write_instrument_read(gap_score__8, 'gap_score__8')
    print('malloc', sys.getsizeof(gap_score__8), 'gap_score__8')
    """
    Computes the local alignment between two sequences using a scoring matrix.
    """
    print(100, 164)
    scoring_matrix__8 = [[(0) for j__8 in range(len(instrument_read(seq2__8,
        'seq2__8')) + 1)] for i__8 in range(len(instrument_read(seq1__8,
        'seq1__8')) + 1)]
    write_instrument_read(scoring_matrix__8, 'scoring_matrix__8')
    print('malloc', sys.getsizeof(scoring_matrix__8), 'scoring_matrix__8')
    print(100, 166)
    traceback_matrix__8 = [[None for j__8 in range(len(instrument_read(
        seq2__8, 'seq2__8')) + 1)] for i__8 in range(len(instrument_read(
        seq1__8, 'seq1__8')) + 1)]
    write_instrument_read(traceback_matrix__8, 'traceback_matrix__8')
    print('malloc', sys.getsizeof(traceback_matrix__8), 'traceback_matrix__8')
    print(100, 168)
    max_score__8 = 0
    write_instrument_read(max_score__8, 'max_score__8')
    print('malloc', sys.getsizeof(max_score__8), 'max_score__8')
    print(100, 169)
    max_pos__8 = None
    write_instrument_read(max_pos__8, 'max_pos__8')
    print('malloc', sys.getsizeof(max_pos__8), 'max_pos__8')
    for i__8 in range(1, len(instrument_read(seq1__8, 'seq1__8')) + 1):
        for j__8 in range(1, len(instrument_read(seq2__8, 'seq2__8')) + 1):
            print(104, 173)
            letter1__8 = instrument_read_sub(instrument_read(seq1__8,
                'seq1__8'), 'seq1__8', instrument_read(i__8, 'i__8') - 1,
                None, None, False)
            write_instrument_read(letter1__8, 'letter1__8')
            print('malloc', sys.getsizeof(letter1__8), 'letter1__8')
            print(104, 174)
            letter2__8 = instrument_read_sub(instrument_read(seq2__8,
                'seq2__8'), 'seq2__8', instrument_read(j__8, 'j__8') - 1,
                None, None, False)
            write_instrument_read(letter2__8, 'letter2__8')
            print('malloc', sys.getsizeof(letter2__8), 'letter2__8')
            if instrument_read(letter1__8, 'letter1__8') == instrument_read(
                letter2__8, 'letter2__8'):
                print(106, 176)
                diagonal_score__8 = instrument_read_sub(instrument_read_sub
                    (instrument_read(scoring_matrix__8, 'scoring_matrix__8'
                    ), 'scoring_matrix__8', instrument_read(i__8, 'i__8') -
                    1, None, None, False), 'scoring_matrix__8[i__8 - 1]', 
                    instrument_read(j__8, 'j__8') - 1, None, None, False
                    ) + instrument_read(match_score__8, 'match_score__8')
                write_instrument_read(diagonal_score__8, 'diagonal_score__8')
                print('malloc', sys.getsizeof(diagonal_score__8),
                    'diagonal_score__8')
            else:
                print(108, 178)
                diagonal_score__8 = instrument_read_sub(instrument_read_sub
                    (instrument_read(scoring_matrix__8, 'scoring_matrix__8'
                    ), 'scoring_matrix__8', instrument_read(i__8, 'i__8') -
                    1, None, None, False), 'scoring_matrix__8[i__8 - 1]', 
                    instrument_read(j__8, 'j__8') - 1, None, None, False
                    ) + instrument_read(mismatch_score__8, 'mismatch_score__8')
                write_instrument_read(diagonal_score__8, 'diagonal_score__8')
                print('malloc', sys.getsizeof(diagonal_score__8),
                    'diagonal_score__8')
            print(107, 180)
            up_score__8 = instrument_read_sub(instrument_read_sub(
                instrument_read(scoring_matrix__8, 'scoring_matrix__8'),
                'scoring_matrix__8', instrument_read(i__8, 'i__8') - 1,
                None, None, False), 'scoring_matrix__8[i__8 - 1]',
                instrument_read(j__8, 'j__8'), None, None, False
                ) + instrument_read(gap_score__8, 'gap_score__8')
            write_instrument_read(up_score__8, 'up_score__8')
            print('malloc', sys.getsizeof(up_score__8), 'up_score__8')
            print(107, 181)
            left_score__8 = instrument_read_sub(instrument_read_sub(
                instrument_read(scoring_matrix__8, 'scoring_matrix__8'),
                'scoring_matrix__8', instrument_read(i__8, 'i__8'), None,
                None, False), 'scoring_matrix__8[i__8]', instrument_read(
                j__8, 'j__8') - 1, None, None, False) + instrument_read(
                gap_score__8, 'gap_score__8')
            write_instrument_read(left_score__8, 'left_score__8')
            print('malloc', sys.getsizeof(left_score__8), 'left_score__8')
            if instrument_read(diagonal_score__8, 'diagonal_score__8'
                ) >= instrument_read(up_score__8, 'up_score__8'):
                if instrument_read(diagonal_score__8, 'diagonal_score__8'
                    ) >= instrument_read(left_score__8, 'left_score__8'):
                    print(115, 185)
                    scoring_matrix__8[instrument_read(instrument_read(i__8,
                        'i__8'), 'i__8')][instrument_read(instrument_read(
                        j__8, 'j__8'), 'j__8')] = instrument_read(
                        diagonal_score__8, 'diagonal_score__8')
                    write_instrument_read_sub(scoring_matrix__8[
                        instrument_read(instrument_read(i__8, 'i__8'),
                        'i__8')],
                        "scoring_matrix__8[instrument_read(i__8, 'i__8')]",
                        instrument_read(instrument_read(j__8, 'j__8'),
                        'j__8'), None, None, False)
                    print(115, 186)
                    traceback_matrix__8[instrument_read(instrument_read(
                        i__8, 'i__8'), 'i__8')][instrument_read(
                        instrument_read(j__8, 'j__8'), 'j__8')] = 'D'
                    write_instrument_read_sub(traceback_matrix__8[
                        instrument_read(instrument_read(i__8, 'i__8'),
                        'i__8')],
                        "traceback_matrix__8[instrument_read(i__8, 'i__8')]",
                        instrument_read(instrument_read(j__8, 'j__8'),
                        'j__8'), None, None, False)
                else:
                    print(117, 188)
                    scoring_matrix__8[instrument_read(instrument_read(i__8,
                        'i__8'), 'i__8')][instrument_read(instrument_read(
                        j__8, 'j__8'), 'j__8')] = instrument_read(left_score__8
                        , 'left_score__8')
                    write_instrument_read_sub(scoring_matrix__8[
                        instrument_read(instrument_read(i__8, 'i__8'),
                        'i__8')],
                        "scoring_matrix__8[instrument_read(i__8, 'i__8')]",
                        instrument_read(instrument_read(j__8, 'j__8'),
                        'j__8'), None, None, False)
                    print(117, 189)
                    traceback_matrix__8[instrument_read(instrument_read(
                        i__8, 'i__8'), 'i__8')][instrument_read(
                        instrument_read(j__8, 'j__8'), 'j__8')] = 'L'
                    write_instrument_read_sub(traceback_matrix__8[
                        instrument_read(instrument_read(i__8, 'i__8'),
                        'i__8')],
                        "traceback_matrix__8[instrument_read(i__8, 'i__8')]",
                        instrument_read(instrument_read(j__8, 'j__8'),
                        'j__8'), None, None, False)
            elif instrument_read(up_score__8, 'up_score__8'
                ) >= instrument_read(left_score__8, 'left_score__8'):
                print(112, 192)
                scoring_matrix__8[instrument_read(instrument_read(i__8,
                    'i__8'), 'i__8')][instrument_read(instrument_read(j__8,
                    'j__8'), 'j__8')] = instrument_read(up_score__8,
                    'up_score__8')
                write_instrument_read_sub(scoring_matrix__8[instrument_read
                    (instrument_read(i__8, 'i__8'), 'i__8')],
                    "scoring_matrix__8[instrument_read(i__8, 'i__8')]",
                    instrument_read(instrument_read(j__8, 'j__8'), 'j__8'),
                    None, None, False)
                print(112, 193)
                traceback_matrix__8[instrument_read(instrument_read(i__8,
                    'i__8'), 'i__8')][instrument_read(instrument_read(j__8,
                    'j__8'), 'j__8')] = 'U'
                write_instrument_read_sub(traceback_matrix__8[
                    instrument_read(instrument_read(i__8, 'i__8'), 'i__8')],
                    "traceback_matrix__8[instrument_read(i__8, 'i__8')]",
                    instrument_read(instrument_read(j__8, 'j__8'), 'j__8'),
                    None, None, False)
            else:
                print(114, 195)
                scoring_matrix__8[instrument_read(instrument_read(i__8,
                    'i__8'), 'i__8')][instrument_read(instrument_read(j__8,
                    'j__8'), 'j__8')] = instrument_read(left_score__8,
                    'left_score__8')
                write_instrument_read_sub(scoring_matrix__8[instrument_read
                    (instrument_read(i__8, 'i__8'), 'i__8')],
                    "scoring_matrix__8[instrument_read(i__8, 'i__8')]",
                    instrument_read(instrument_read(j__8, 'j__8'), 'j__8'),
                    None, None, False)
                print(114, 196)
                traceback_matrix__8[instrument_read(instrument_read(i__8,
                    'i__8'), 'i__8')][instrument_read(instrument_read(j__8,
                    'j__8'), 'j__8')] = 'L'
                write_instrument_read_sub(traceback_matrix__8[
                    instrument_read(instrument_read(i__8, 'i__8'), 'i__8')],
                    "traceback_matrix__8[instrument_read(i__8, 'i__8')]",
                    instrument_read(instrument_read(j__8, 'j__8'), 'j__8'),
                    None, None, False)
            if instrument_read_sub(instrument_read_sub(instrument_read(
                scoring_matrix__8, 'scoring_matrix__8'),
                'scoring_matrix__8', instrument_read(i__8, 'i__8'), None,
                None, False), 'scoring_matrix__8[i__8]', instrument_read(
                j__8, 'j__8'), None, None, False) >= instrument_read(
                max_score__8, 'max_score__8'):
                print(118, 199)
                max_score__8 = instrument_read_sub(instrument_read_sub(
                    instrument_read(scoring_matrix__8, 'scoring_matrix__8'),
                    'scoring_matrix__8', instrument_read(i__8, 'i__8'),
                    None, None, False), 'scoring_matrix__8[i__8]',
                    instrument_read(j__8, 'j__8'), None, None, False)
                write_instrument_read(max_score__8, 'max_score__8')
                print('malloc', sys.getsizeof(max_score__8), 'max_score__8')
                print(118, 200)
                max_pos__8 = instrument_read(i__8, 'i__8'), instrument_read(
                    j__8, 'j__8')
                write_instrument_read(max_pos__8, 'max_pos__8')
                print('malloc', sys.getsizeof(max_pos__8), 'max_pos__8')
    print(103, 202)
    i__8 = instrument_read(max_pos__8, 'max_pos__8')
    write_instrument_read(i__8, 'i__8')
    print('malloc', sys.getsizeof(i__8), 'i__8')
    print(103, 203)
    j__8 = instrument_read(max_pos__8, 'max_pos__8')
    write_instrument_read(j__8, 'j__8')
    print('malloc', sys.getsizeof(j__8), 'j__8')
    print(103, 204)
    aln1__8 = ''
    write_instrument_read(aln1__8, 'aln1__8')
    print('malloc', sys.getsizeof(aln1__8), 'aln1__8')
    print(103, 205)
    aln2__8 = ''
    write_instrument_read(aln2__8, 'aln2__8')
    print('malloc', sys.getsizeof(aln2__8), 'aln2__8')
    while instrument_read_sub(instrument_read_sub(instrument_read(
        traceback_matrix__8, 'traceback_matrix__8'), 'traceback_matrix__8',
        instrument_read(i__8, 'i__8'), None, None, False),
        'traceback_matrix__8[i__8]', instrument_read(j__8, 'j__8'), None,
        None, False) != None:
        if instrument_read_sub(instrument_read_sub(instrument_read(
            traceback_matrix__8, 'traceback_matrix__8'),
            'traceback_matrix__8', instrument_read(i__8, 'i__8'), None,
            None, False), 'traceback_matrix__8[i__8]', instrument_read(j__8,
            'j__8'), None, None, False) == 'D':
            print(123, 208)
            aln1__8 += instrument_read_sub(instrument_read(seq1__8,
                'seq1__8'), 'seq1__8', instrument_read(i__8, 'i__8') - 1,
                None, None, False)
            write_instrument_read(aln1__8, 'aln1__8')
            print(123, 209)
            aln2__8 += instrument_read_sub(instrument_read(seq2__8,
                'seq2__8'), 'seq2__8', instrument_read(j__8, 'j__8') - 1,
                None, None, False)
            write_instrument_read(aln2__8, 'aln2__8')
            print(123, 210)
            i__8 -= 1
            write_instrument_read(i__8, 'i__8')
            print(123, 211)
            j__8 -= 1
            write_instrument_read(j__8, 'j__8')
        elif instrument_read_sub(instrument_read_sub(instrument_read(
            traceback_matrix__8, 'traceback_matrix__8'),
            'traceback_matrix__8', instrument_read(i__8, 'i__8'), None,
            None, False), 'traceback_matrix__8[i__8]', instrument_read(j__8,
            'j__8'), None, None, False) == 'L':
            print(126, 213)
            aln1__8 += '-'
            write_instrument_read(aln1__8, 'aln1__8')
            print(126, 214)
            aln2__8 += instrument_read_sub(instrument_read(seq2__8,
                'seq2__8'), 'seq2__8', instrument_read(j__8, 'j__8') - 1,
                None, None, False)
            write_instrument_read(aln2__8, 'aln2__8')
            print(126, 215)
            j__8 -= 1
            write_instrument_read(j__8, 'j__8')
        elif instrument_read_sub(instrument_read_sub(instrument_read(
            traceback_matrix__8, 'traceback_matrix__8'),
            'traceback_matrix__8', instrument_read(i__8, 'i__8'), None,
            None, False), 'traceback_matrix__8[i__8]', instrument_read(j__8,
            'j__8'), None, None, False) == 'U':
            print(129, 217)
            aln1__8 += instrument_read_sub(instrument_read(seq1__8,
                'seq1__8'), 'seq1__8', instrument_read(i__8, 'i__8') - 1,
                None, None, False)
            write_instrument_read(aln1__8, 'aln1__8')
            print(129, 218)
            aln2__8 += '-'
            write_instrument_read(aln2__8, 'aln2__8')
            print(129, 219)
            i__8 -= 1
            write_instrument_read(i__8, 'i__8')
    print('exit scope 8')
    return instrument_read_sub(instrument_read(aln1__8, 'aln1__8'),
        'aln1__8', None, None, None, True), instrument_read_sub(instrument_read
        (aln2__8, 'aln2__8'), 'aln2__8', None, None, None, True)
    print('exit scope 8')


def main():
    print('enter scope 9')
    print(1, 222)
    print(134, 223)
    fasta_file__9 = (
        '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/fasta_example.fasta'
        )
    write_instrument_read(fasta_file__9, 'fasta_file__9')
    print('malloc', sys.getsizeof(fasta_file__9), 'fasta_file__9')
    print(134, 224)
    fastq_file__9 = (
        '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/fastq_example.fastq'
        )
    write_instrument_read(fastq_file__9, 'fastq_file__9')
    print('malloc', sys.getsizeof(fastq_file__9), 'fastq_file__9')
    print(134, 225)
    output_file__9 = (
        '/Users/PatrickMcEwen/git_container/codesign/src/benchmarks/supplemental_files/output_file.txt'
        )
    write_instrument_read(output_file__9, 'output_file__9')
    print('malloc', sys.getsizeof(output_file__9), 'output_file__9')
    darwin_wga_workflow(instrument_read(fasta_file__9, 'fasta_file__9'),
        instrument_read(fastq_file__9, 'fastq_file__9'), instrument_read(
        output_file__9, 'output_file__9'))
    print('exit scope 9')


if instrument_read(__name__, '__name__') == '__main__':
    main()
