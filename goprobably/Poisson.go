package goprobably

import (
	G "gorgonia.org/gorgonia"
	_"fmt"
	"math"
	"log"
)
type Poisson struct {
	Lambda *G.Node
	Dtype string
}

func (s Poisson) GLogProba(g *G.ExprGraph, xG *G.Node) *G.Node {
	LogLambda := G.Must(G.Log(s.Lambda))

	LambdaKG := G.Must(G.Mul(LogLambda, xG))

	LambdaKExpNegLambdaG := G.Must(G.Sub(LambdaKG, s.Lambda))

	//TODO: Gamma function realisation
	K, ok := xG.Value().Data().(float64)
	if !ok {log.Fatal("K is not float64")}
	LogGammaK, _ := math.Lgamma(K)

	LogPdfG := G.Must(G.Sub(LambdaKExpNegLambdaG, G.NewConstant(LogGammaK)))

	return LogPdfG
}

func NewPoisson(Lambda *G.Node) Distribution {
	return Poisson{Lambda: Lambda, Dtype: "discrete"}
}