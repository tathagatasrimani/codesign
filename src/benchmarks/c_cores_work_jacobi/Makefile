CATAPULT = catapult

gui:
	$(CATAPULT) build.ccs

# test/MatMult.v1:
# 	$(CATAPULT) -shell -file scripts/run_c_test.tcl

# c_test: test/MatMult.v1
# 	cd test/MatMult.v1 && make -f scverify/Verify_orig_cxx_osci.mk sim

build/Jacobi2D.v1:
	$(CATAPULT) -shell -file scripts/jacobi2d.tcl

build_design: build/Jacobi2D.v1


clean:
	rm -rf build.ccs
	rm -rf build
	rm -rf test*