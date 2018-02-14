package goprobably

import (
	G "gorgonia.org/gorgonia"
	_"fmt"
)

const EPS = 1e-124

type Distribution interface {
	GLogProba(g *G.ExprGraph, x *G.Node) *G.Node
}
