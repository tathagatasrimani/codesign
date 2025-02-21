CATAPULT = /cad/mentor/2019.11/Catapult_Synthesis_10.4b-841621/Mgc_home/bin/catapult

gui:
	$(CATAPULT) build.ccs

test/MatMult.v1:
	$(CATAPULT) -shell -file scripts/run_c_test.tcl

c_test: test/MatMult.v1
	cd test/MatMult.v1 && make -f scverify/Verify_orig_cxx_osci.mk sim

build/MatMult.v1:
	$(CATAPULT) -shell -file scripts/matmult.tcl

build_design: build/MatMult.v1


clean:
	rm -rf build.ccs
	rm -rf build
	rm -rf test*