package goprobably

import (
	G "gorgonia.org/gorgonia"
	_"fmt"
	_"gonum.org/v1/gonum/mat"
	"math"
)

type LogNormal struct {
	Mean, Sigma *G.Node
	Dtype string
}

func (s LogNormal) GLogProba(g *G.ExprGraph, xG *G.Node) *G.Node {
	xLogG := G.Must(G.Log(xG))

	xSubMeanG := G.Must(G.Sub(xLogG, s.Mean))

	invSigmaG := G.Must(G.Inverse(s.Sigma))

	xSubMeaninvSigmaG := G.Must(G.Mul(invSigmaG, xSubMeanG))

	xSubMeaninvSigmaxSubMeanVecG := G.Must(G.Mul(xSubMeaninvSigmaG, xSubMeanG))

	xSubMeaninvSigmaxSubMeanG := G.Must(G.Sum(xSubMeaninvSigmaxSubMeanVecG))

	xSubMeaninvSigmaxSubMean2G := G.Must(G.Div(xSubMeaninvSigmaxSubMeanG, G.NewConstant(-2.)))

	// part with log(1 / sqrt(2 pi sigma))
	DetSigma := G.Must(G.Div(G.NewConstant(math.Pi * 2), s.Sigma))

	LogDetSigma := G.Must(G.Log(DetSigma))

	LogDetSigma2 := G.Must(G.Div(LogDetSigma, G.NewConstant(2.)))

	xSubMeaninvSigmaxSubMean2Sub2PiDetSigmaG := G.Must(G.Sub(xSubMeaninvSigmaxSubMean2G, LogDetSigma2))

	xSubMeaninvSigmaxSubMean2Sub2PiDetSigmaDivXG := G.Must(G.Sub(xSubMeaninvSigmaxSubMean2Sub2PiDetSigmaG, xLogG))

	return xSubMeaninvSigmaxSubMean2Sub2PiDetSigmaDivXG
}

func NewLogNormal(Mean, Sigma *G.Node) Distribution {
	return LogNormal{Mean: Mean, Sigma: Sigma, Dtype: "continuous"}
}