CATAPULT = /cad/mentor/2024.2/Mgc_home/bin/catapult

gui:
	$(CATAPULT) build.ccs

test/basic_aes.v1:
	$(CATAPULT) -shell -file scripts/run_c_test.tcl

c_test: test/basic_aes.v1
	cd test/basic_aes.v1 && make -f scverify/Verify_orig_cxx_osci.mk sim

build/basic_aes.v1:
	$(CATAPULT) -shell -file scripts/basic_aes.tcl

build_design: build/basic_aes.v1


clean:
	rm -rf build.ccs
	rm -rf build
	rm -rf test*