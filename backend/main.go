package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

type SimRequest struct {
	Neurons   int     `json:"n_neurons"`
	TEnd      float64 `json:"t_end"`
	DT        float64 `json:"dt"`
	NParam    float64 `json:"n_param"`
	BaseGamma float64 `json:"base_gamma"`
	DCouple   float64 `json:"d_couple"`
	Alpha     float64 `json:"alpha"`
	Seed      int     `json:"seed"`
}

type ScanRequest struct {
	Neurons   int     `json:"n_neurons"`
	TEnd      float64 `json:"t_end"`
	DT        float64 `json:"dt"`
	NParam    float64 `json:"n_param"`
	BaseGamma float64 `json:"base_gamma"`

	ScanType string  `json:"scan_type"`
	DMin     float64 `json:"d_min"`
	DMax     float64 `json:"d_max"`
	DSteps   int     `json:"d_steps"`

	AlphaMin   float64 `json:"alpha_min"`
	AlphaMax   float64 `json:"alpha_max"`
	AlphaSteps int     `json:"alpha_steps"`

	DeltaMin   float64 `json:"delta_min"`
	DeltaMax   float64 `json:"delta_max"`
	DeltaSteps int     `json:"delta_steps"`
	FixedAlpha float64 `json:"fixed_alpha"`

	Seed int `json:"seed"`
}

func proxyToPython(w http.ResponseWriter, r *http.Request, endpoint string) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	pyURL := "http://localhost:8000" + endpoint
	client := &http.Client{Timeout: 600 * time.Second}

	resp, err := client.Post(pyURL, "application/json", bytes.NewBuffer(body))
	if err != nil {
		log.Printf("Python Core Error: %v", err)
		http.Error(w, "Core unavailable", http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "application/json")
	io.Copy(w, resp.Body)
}

func handleSimulate(w http.ResponseWriter, r *http.Request) {
	proxyToPython(w, r, "/simulate")
}

func handleScan(w http.ResponseWriter, r *http.Request) {
	proxyToPython(w, r, "/scan")
}

func main() {
	http.HandleFunc("/api/run", handleSimulate)
	http.HandleFunc("/api/scan", handleScan)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	fmt.Println("Go Orchestrator running on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}
