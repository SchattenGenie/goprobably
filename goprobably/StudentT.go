package goprobably

import (
	G "gorgonia.org/gorgonia"
	_"fmt"
)

type StudentT struct {
	nu    *G.Node
	Dtype string
}

func (s StudentT) GLogProba(g *G.ExprGraph, x *G.Node) *G.Node {
	return G.NewConstant(2.)
}