package goprobably

import (
	_ "fmt"
	G "gorgonia.org/gorgonia"
)

type Exponential struct {
	Lambda *G.Node
	Dtype  string
}

// Step function is not differentiable yet in Gorgonia
// so we I an using sigmoid as its approximation
func (s Exponential) GLogProba(g *G.ExprGraph, xG *G.Node) *G.Node {
	xMulG := G.Must(G.Div(xG, G.NewConstant(EPS)))

	// Approximation of sign(x)
	xSignG := G.Must(G.Sigmoid(xMulG))

	// log(sign(x))
	xSignLogG := G.Must(G.Log(xSignG))

	// -x
	NegxG := G.Must(G.Neg(xG))

	// -x / lambda
	NegxDivLambdaG := G.Must(G.Div(NegxG, s.Lambda))

	// log(lambda)
	LogLambdaG := G.Must(G.Log(s.Lambda))

	// - x / lambda - log(lambda)
	LogResultG := G.Must(G.Sub(NegxDivLambdaG, LogLambdaG))

	// (- x / lambda - log(lambda)) + log(sign(x))
	LogResultSignG := G.Must(G.Add(LogResultG, xSignLogG))

	return LogResultSignG
}

func NewExponential(Lambda *G.Node) Distribution {
	return Exponential{Lambda: Lambda, Dtype: "continuous"}
}
