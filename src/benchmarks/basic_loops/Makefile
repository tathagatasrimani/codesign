CATAPULT = /cad/mentor/2019.11/Catapult_Synthesis_10.4b-841621/Mgc_home/bin/catapult

gui:
	$(CATAPULT) build.ccs

build/BasicLoops.v1:
	$(CATAPULT) -shell -file scripts/basic_loops.tcl

build_design: build/BasicLoops.v1

clean:
	rm -rf build.ccs
	rm -rf build
	rm -rf test*