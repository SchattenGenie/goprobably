package goprobably

import (
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
	"log"
	_"reflect"
	"math"
	"math/rand"
)
var err error
type MC struct {
	Step float64
	Length int
	Variables []*G.Node // p(var | opt)
	Distributions []Distribution // Distributions with logpdf
	VariablesPreviousStep []*T.Dense // Variables on previous step
	PPreviousStep []*T.Dense // momentum on previous step
	LogPdf *G.Node
	Grad G.Nodes
	P []*G.Node // momentum generator
	G *G.ExprGraph
	Std float64
}

type HMC interface {
	Init(g *G.ExprGraph)
	Generate() []*T.Dense
}

func (h *MC) Init(g *G.ExprGraph) {
	LogPdf := G.NewScalar(g, G.Float64, G.WithName("LogPdf"))
	G.Let(LogPdf, 0.)

	// Create LogPdf variable
	for i := range h.Distributions {
		LogPdf, err = G.Add(h.Distributions[i].GLogProba(g, h.Variables[i]), LogPdf)
		if err != nil {log.Fatal(err)}
	}
	h.LogPdf = LogPdf
	// Gradients
	Grad, err := G.Grad(LogPdf, h.Variables...)
	if err != nil {log.Fatal(err)}
	h.Grad = Grad
	h.G = g
}

func (h *MC) Generate(machine G.VM) []*T.Dense {
	// Reset don't null variables. Only reset graph execution
	// Calculate stats for previous step: HOld, LogPdf and sum(p**2/std) / 2
	machine.Reset()
	for i := range h.Variables {
		if h.VariablesPreviousStep[i].IsScalar() {
			G.Let(h.Variables[i], h.VariablesPreviousStep[i].Data())
		} else {
			G.Let(h.Variables[i], h.VariablesPreviousStep[i])
		}
	}
	machine.RunAll()

	LogPdf, ok := h.LogPdf.Value().Data().(float64)
	if !ok {log.Fatal("LogPdf is not float64")}

	PSum2 := 0.
	for i := range h.Variables {
		Square, err := T.Square(h.PPreviousStep[i])
		SquareDivStd, err := T.Div(Square, h.Std * h.Std)
		SumSquareDivStd, err := T.Sum(SquareDivStd)
		if err != nil {log.Fatal(err)}
		sumFloat64, ok := SumSquareDivStd.ScalarValue().(float64)
		if !ok {log.Fatal("SumSquareDivStd is not float64")}
		PSum2 = PSum2 + sumFloat64 / 2.
	}

	HOld := LogPdf + PSum2

	// Get new P
	var PNextStep []*T.Dense
	for i := range h.Variables {
		p := h.P[i].Value()
		pValue, ok := p.Data().(float64)
		if !ok {log.Fatal("pValue is not float64")}
		pT := T.NewDense(T.Float64, p.Shape(), T.WithBacking([]float64{pValue}))
		PNextStep = append(PNextStep, pT)
	}
	// Init new Variables
	var VariablesNextStep []*T.Dense
	for i := range h.Variables {
		Var := T.NewDense(T.Float64, h.Variables[i].Shape())
		h.VariablesPreviousStep[i].CopyTo(Var)
		VariablesNextStep = append(VariablesNextStep, Var)
	}

	// p = p - step * grad / 2
	for i := range h.Variables {
		Grad := h.Grad[i].Value()
		GradValue, ok := Grad.Data().(float64)
		if !ok {log.Fatal("GradValue is not float64")}
		// scalar and tensor differently
		GradT := T.NewDense(T.Float64, Grad.Shape(), T.WithBacking([]float64{GradValue}))

		Grad2StepT, err := GradT.MulScalar(h.Step / 2., true)
		if err != nil {log.Fatal(err)}
		PNextStep[i], err = PNextStep[i].Add(Grad2StepT)
	}

	// VariablesNextStep = VariablesNextStep + step * p / std
	for i := range h.Variables {
		Add, err := PNextStep[i].MulScalar(h.Step / h.Std, true)
		if err != nil {log.Fatal(err)}
		VariablesNextStep[i], err = VariablesNextStep[i].Add(Add)
		if err != nil {log.Fatal(err)}
	}

	// p = p - step * grad / 2
	// Need new grads
	machine.Reset()
	for i := range h.Variables {
		if h.VariablesPreviousStep[i].IsScalar() {
			G.Let(h.Variables[i], VariablesNextStep[i].Data())
		} else {
			G.Let(h.Variables[i], VariablesNextStep[i])
		}
	}
	machine.RunAll()

	// p = p - step * grad / 2
	for i := range h.Variables {
		Grad := h.Grad[i].Value()
		GradValue, ok := Grad.Data().(float64)
		if !ok {log.Fatal("GradValue is not float64")}
		// scalar and tensor differently
		GradT := T.NewDense(T.Float64, Grad.Shape(), T.WithBacking([]float64{GradValue}))

		Grad2StepT, err := GradT.MulScalar(h.Step / 2., true)
		if err != nil {log.Fatal(err)}
		PNextStep[i], err = PNextStep[i].Add(Grad2StepT)
	}


	LogPdf, ok = h.LogPdf.Value().Data().(float64)
	if !ok {log.Fatal("LogPdf is not float64")}
	PSum2 = 0.
	for i := range h.Variables {
		Square, err := T.Square(h.PPreviousStep[i])
		SquareDivStd, err := T.Div(Square, h.Std)
		SumSquareDivStd, err := T.Sum(SquareDivStd)
		if err != nil {log.Fatal(err)}
		sumFloat64, ok := SumSquareDivStd.ScalarValue().(float64)
		if !ok {log.Fatal("SumSquareDivStd is not float64")}
		PSum2 = PSum2 + sumFloat64 / 2.
	}

	HNew := LogPdf + PSum2

	RejectProba := math.Min(1,math.Exp(HNew - HOld))

	if rand.Float64() < RejectProba {
		h.VariablesPreviousStep = VariablesNextStep
		h.PPreviousStep = PNextStep
		return h.VariablesPreviousStep
	}

	return h.VariablesPreviousStep
}