CATAPULT = /cad/mentor/2024.2/Mgc_home/bin/catapult

gui:
	$(CATAPULT) build.ccs

test/matmult.v1:
	$(CATAPULT) -shell -file scripts/run_c_test.tcl

c_test: test/matmult.v1
	cd test/matmult.v1 && make -f scverify/Verify_orig_cxx_osci.mk sim

build/matmult.v1:
	$(CATAPULT) -shell -file scripts/matmult.tcl

build_design: build/matmult.v1


clean:
	rm -rf build.ccs
	rm -rf build
	rm -rf test*