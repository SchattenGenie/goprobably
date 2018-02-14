package main

import (
	P "./goprobably"
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
	"log"
	_"reflect"
	"fmt"
	"time"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	g := G.NewGraph()

	Mean1 := G.NewScalar(g, G.Float64, G.WithName("Mean"))
	Sigma1 := G.NewScalar(g, G.Float64, G.WithName("Sigma"))
	x1 := G.NewScalar(g, G.Float64, G.WithShape(), G.WithName("x1"))
	Distribution1 := P.NewNormal(Mean1, Sigma1)
	xT1 := T.New(T.WithShape(), T.WithBacking([]float64{1.}))
	pT1 := T.New(T.WithShape(), T.WithBacking([]float64{1.}))


	Lambda2 := G.NewScalar(g, G.Float64, G.WithName("Lambda"))
	x2 := G.NewScalar(g, G.Float64, G.WithShape(), G.WithName("x2"))
	Distribution2 := P.Exponential{Lambda:Lambda2}
	xT2 := T.New(T.WithShape(), T.WithBacking([]float64{1.}))
	pT2 := T.New(T.WithShape(), T.WithBacking([]float64{1.}))

	Std := 1.
	hmc := P.MC{
		Step: .1,
		Length: 1,
		Variables: []*G.Node{x1, x2},
		Distributions: []P.Distribution{Distribution1, Distribution2},
		VariablesPreviousStep:[]*T.Dense{xT1, xT2},
		PPreviousStep:[]*T.Dense{pT1, pT2},
		P: []*G.Node{G.GaussianRandomNode(g, G.Float64, 0., Std), G.GaussianRandomNode(g, G.Float64, 0., Std)},
		Std: Std,
	}

	hmc.Init(g)
	machine := G.NewTapeMachine(g)
	G.Let(Mean1, 1000.)
	G.Let(Sigma1, .01)
	G.Let(Lambda2, 5.)

	start := time.Now()
	for i := 1; i <= 1000; i++ {
		fmt.Println(hmc.Generate(machine))
	}
	elapsed := time.Since(start)
	log.Printf("Generation took %s", elapsed)
	fmt.Println('n')
	return
}
