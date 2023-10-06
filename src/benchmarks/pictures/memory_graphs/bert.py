graph {
	657 [label=and
]
	658 [label=or
]
	659 [label="+
"]
	660 [label="+
"]
	661 [label="+
"]
	662 [label="+
"]
	663 [label="+
"]
	664 [label="+
"]
	665 [label="+
"]
	666 [label="+
"]
	667 [label="+
"]
	668 [label="+
"]
	669 [label="+
"]
	670 [label="+
"]
	671 [label="+
"]
	672 [label="+
"]
	673 [label="+
"]
	674 [label="-
"]
	675 [label="-
"]
	676 [label="-
"]
	677 [label="-
"]
	678 [label="-
"]
	679 [label="-
"]
	680 [label="-
"]
	681 [label="-
"]
	682 [label="-
"]
	683 [label="-
"]
	684 [label="-
"]
	685 [label="-
"]
	686 [label="-
"]
	687 [label="-
"]
	688 [label="-
"]
	689 [label="*
"]
	690 [label="*
"]
	691 [label="*
"]
	692 [label="*
"]
	693 [label="*
"]
	694 [label="*
"]
	695 [label="*
"]
	696 [label="*
"]
	697 [label="*
"]
	698 [label="*
"]
	699 [label="*
"]
	700 [label="*
"]
	701 [label="*
"]
	702 [label="*
"]
	703 [label="*
"]
	704 [label="//
"]
	705 [label="//
"]
	706 [label="//
"]
	707 [label="//
"]
	708 [label="//
"]
	709 [label="//
"]
	710 [label="//
"]
	711 [label="//
"]
	712 [label="//
"]
	713 [label="//
"]
	714 [label="//
"]
	715 [label="//
"]
	716 [label="//
"]
	717 [label="//
"]
	718 [label="//
"]
	719 [label="%
"]
	720 [label="<<
"]
	721 [label=">>
"]
	722 [label="|
"]
	723 [label="^
"]
	724 [label="&
"]
	725 [label="==
"]
	726 [label="!=
"]
	727 [label="<
"]
	728 [label="<=
"]
	729 [label=">
"]
	730 [label=">=
"]
	731 [label="!=
"]
	732 [label="-
"]
	733 [label="+
"]
	734 [label="!
"]
	735 [label="~
"]
	736 [label="d_k
location: 32
Write"]
	736 -- 704 [label="size: 32"]
	689 -- 704 [label=""]
	737 [label="embeddings
location: 64
Read"]
	737 -- 659 [label="size: 896"]
	732 -- 689 [label=""]
	929 [label="arr
location: 1056
Read"]
	929 -- 689 [label="size: 1296"]
	931 [label="neg
location: 2352
Read"]
	931 -- 734 [label="size: 32"]
	2223 [label="neg
location: 960
Read"]
	2223 -- 734 [label="size: 32"]
	2225 [label="embeddings
location: 64
Read"]
	2225 -- 689 [label="size: 896"]
	2226 [label="W_V
location: 4976
Read"]
	2226 -- 689 [label="size: 1296"]
	2227 [label="embeddings
location: 64
Read"]
	2227 -- 690 [label="size: 896"]
	2228 [label="W_Q
location: 2384
Read"]
	2228 -- 690 [label="size: 1296"]
	2229 [label="embeddings
location: 64
Read"]
	2229 -- 691 [label="size: 896"]
	2230 [label="W_K
location: 3680
Read"]
	2230 -- 691 [label="size: 1296"]
	689 -- 659 [label=""]
	2231 [label="sumV
location: 960
Read"]
	2231 -- 659 [label="size: 24"]
	690 -- 660 [label=""]
	691 -- 661 [label=""]
	11441 [label="Q
location: 1056
Read"]
	11441 -- 689 [label="size: 912"]
	11442 [label="K
location: 4976
Read"]
	11442 -- 689 [label="size: 912"]
	11443 [label="scores
location: 7184
Read"]
	11443 -- 659 [label="size: 640"]
	11697 [label="scores
location: 7184
Read"]
	11697 -- 704 [label="size: 640"]
	11729 [label="sum_val
location: 960
Read"]
	11729 -- 704 [label="size: 24"]
	11730 [label="result
location: 2080
Write"]
	11730 -- 704 [label="size: 176"]
	12081 [label="scores
location: 7184
Read"]
	12081 -- 689 [label="size: 640"]
	12082 [label="V
location: 6272
Read"]
	12082 -- 689 [label="size: 912"]
	12083 [label="out
location: 3280
Read"]
	12083 -- 659 [label="size: 192"]
	22193 [label="multi_head_out
location: 3680
Read"]
	22193 -- 689 [label="size: 896"]
	22194 [label="W_attn
location: 8992
Read"]
	22194 -- 689 [label="size: 1280"]
	26801 [label="src
location: 11168
Read"]
	26801 -- 659 [label="size: 896"]
	26802 [label="dst
location: 10272
Read"]
	26802 -- 659 [label="size: 896"]
	27089 [label="row
location: 1968
Read"]
	27089 -- 674 [label="size: 112"]
	27090 [label="result
location: 1024
Read"]
	27090 -- 659 [label="size: 24"]
	27092 [label="row
location: 1968
Read"]
	27092 -- 659 [label="size: 112"]
	27128 [label="diff
location: 2080
Read"]
	27128 -- 689 [label="size: 32"]
	27131 [label="diff
location: 2080
Write"]
	27131 -- 674 [label="size: 32"]
	27596 [label="dev
location: 2112
Read"]
	27596 -- 725 [label="size: 24"]
	27633 [label="arr
location: 12064
Read"]
	27633 -- 674 [label="size: 896"]
	27634 [label="mean
location: 2136
Read"]
	27634 -- 674 [label="size: 32"]
	674 -- 704 [label=""]
	27635 [label="dev
location: 2112
Read"]
	27635 -- 704 [label="size: 24"]
	27636 [label="weights
location: 5888
Read"]
	27636 -- 689 [label="size: 208"]
	27637 [label="biases
location: 4576
Read"]
	27637 -- 659 [label="size: 208"]
	27638 [label="arr
location: 12064
Write"]
	27638 -- 659 [label="size: 896"]
	32045 [label="arr
location: 4784
Read"]
	32045 -- 689 [label="size: 112"]
	32046 [label="W
location: 12960
Read"]
	32046 -- 689 [label="size: 4736"]
	32047 [label="sum_val
location: 2168
Read"]
	32047 -- 659 [label="size: 32"]
	69202 [label="result
location: 17696
Read"]
	69202 -- 659 [label="size: 496"]
	69745 [label="arr
location: 4784
Read"]
	69745 -- 674 [label="size: 112"]
	69750 [label="arr
location: 4784
Write"]
	69750 -- 659 [label="size: 112"]
	74445 [label="sum_val
location: 2168
Read"]
	74445 -- 704 [label="size: 32"]
	74446 [label="result
location: 17696
Write"]
	74446 -- 704 [label="size: 496"]
}
