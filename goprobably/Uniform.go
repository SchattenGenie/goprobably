package goprobably

import (
	G "gorgonia.org/gorgonia"
	_"fmt"
	"log"
)

type Uniform struct {
	A, B *G.Node
}

// Step function is not yet differentiable in Gorgonia
// so I an using sigmoid as its approximation
func (s Uniform) GLogProba(g *G.ExprGraph, xG *G.Node) *G.Node {
	diffxaG, err := G.Sub(xG, s.A)
	if err != nil {log.Fatal(err)}
	diffxaMulG, err := G.Div(diffxaG, G.NewConstant(EPS))
	if err != nil {log.Fatal(err)}

	diffbxG, err := G.Sub(s.B, xG)
	if err != nil {log.Fatal(err)}
	diffbxMulG, err := G.Div(diffbxG, G.NewConstant(EPS))
	if err != nil {log.Fatal(err)}

	diffxaSignG, err := G.Sigmoid(diffxaMulG)
	if err != nil {log.Fatal(err)}
	diffbxSignG, err := G.Sigmoid(diffbxMulG)
	if err != nil {log.Fatal(err)}

	MultipliedG, err := G.Mul(diffxaSignG, diffbxSignG)
	if err != nil {log.Fatal(err)}

	supportG, err := G.Sub(s.B, s.A)
	if err != nil {log.Fatal(err)}

	NormedG, err := G.Div(MultipliedG, supportG)
	if err != nil {log.Fatal(err)}

	LogG, err := G.Log(NormedG)
	if err != nil {log.Fatal(err)}

	return LogG
}