package test

import (
	_ "fmt"
	_ "math/rand"
	_ "time"
	"fmt"
	P "./goprobably"
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
	"log"
	_"reflect"
)

func measure(g *G.ExprGraph, distr P.Distribution, xG *G.Node) *G.Node {
	return distr.GLogProba(g, xG)
}

func test() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	g := G.NewGraph()

	Mean := G.NewVector(g, G.Float64, G.WithShape(2), G.WithName("Mean"))
	Sigma := G.NewMatrix(g, G.Float64, G.WithShape(2, 2), G.WithName("Sigma"))
	InvSigma := G.NewMatrix(g, G.Float64, G.WithShape(2, 2), G.WithName("InvSigma"))

	distr := P.MultivariateNormal{Mean:Mean, Sigma:Sigma, InvSigma:InvSigma}

	xG := G.NewVector(g, G.Float64, G.WithShape(2), G.WithName("x"))

	MeanT := T.New(T.WithShape(2), T.WithBacking([]float64{2., 3.}))
	xT := T.New(T.WithShape(2), T.WithBacking([]float64{3., 3.}))
	SigmaT := T.New(T.WithShape(2, 2), T.WithBacking([]float64{2., 3., 4., 5.}))
	invSigmaT, err := T.Inv(SigmaT)


	G.Let(Mean, MeanT)
	G.Let(Sigma, SigmaT)
	G.Let(xG, xT)
	G.Let(InvSigma, invSigmaT)

	answer := measure(g, distr, xG)
	grad, err := G.Grad(answer, xG)

	machine := G.NewTapeMachine(g)


	err = machine.RunAll()
	if err != nil {log.Fatal(err)}


	fmt.Println(answer.Value())
	fmt.Println(grad[0].Value())
}
