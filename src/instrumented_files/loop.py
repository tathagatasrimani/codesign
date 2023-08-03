class loop:
    start_unroll = None
    stop_unroll = None
    def pattern_seek(self, max_unroll=1):
        print("pattern_seek_" + str(max_unroll))