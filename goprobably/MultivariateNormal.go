package goprobably

import (
	G "gorgonia.org/gorgonia"
	_"fmt"
	"log"
	"gonum.org/v1/gonum/mat"
	"math"
)

type MultivariateNormal struct {
	Mean, Sigma, InvSigma *G.Node
}


// There is no Det and Inv in Gorgonia yet
// thats why we can't make inference on those types
// TODO: make G.MatrixDet, G.MatrixInv

func (s MultivariateNormal) GLogProba(g *G.ExprGraph, xG *G.Node) *G.Node {
	Shape, err := s.Mean.Shape().DimSize(0)
	if err != nil {log.Fatal(err)}

	xSubMeanG, err := G.Sub(xG, s.Mean)
	if err != nil {log.Fatal(err)}


	SigmaSlice, ok := s.Sigma.Value().Data().([]float64)
	if !ok {log.Fatal("Sigma is not []float64")}
	SigmaD := mat.NewDense(Shape, Shape, SigmaSlice)
	/**
	SigmaInvD := mat.NewDense(Shape, Shape, nil)
	SigmaInvD.Inverse(SigmaD)
	**/

	xSubMeaninvSigmaG, err := G.Mul(s.InvSigma, xSubMeanG)
	if err != nil {log.Fatal(err)}

	xSubMeaninvSigmaxSubMeanVecG, err := G.Mul(xSubMeaninvSigmaG, xSubMeanG)
	if err != nil {log.Fatal(err)}

	xSubMeaninvSigmaxSubMeanG, err := G.Sum(xSubMeaninvSigmaxSubMeanVecG)
	if err != nil {log.Fatal(err)}

	xSubMeaninvSigmaxSubMean2G, err := G.Div(xSubMeaninvSigmaxSubMeanG, G.NewConstant(-2.))
	if err != nil {log.Fatal(err)}

	// part with log(1 / sqrt(2 pi sigma))
	DetSigma := math.Abs(mat.Det(SigmaD))

	xSubMeaninvSigmaxSubMean2Sub2PiDetSigmaG, err := G.Sub(xSubMeaninvSigmaxSubMean2G, G.NewConstant(1. / (2. * math.Log(2. * math.Pi * DetSigma))))
	if err != nil {log.Fatal(err)}

	return xSubMeaninvSigmaxSubMean2Sub2PiDetSigmaG
}