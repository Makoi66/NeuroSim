package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

func writeCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

func proxyToPython(w http.ResponseWriter, r *http.Request, endpoint string) {
	writeCORS(w)

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

	// Forward upstream headers (excluding hop-by-hop and CORS, which we set ourselves).
	for k, vs := range resp.Header {
		if k == "Access-Control-Allow-Origin" || k == "Access-Control-Allow-Methods" || k == "Access-Control-Allow-Headers" {
			continue
		}
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	if _, err := io.Copy(w, resp.Body); err != nil {
		log.Printf("Proxy copy error (%s): %v", endpoint, err)
	}
}

func handleSimulate(w http.ResponseWriter, r *http.Request) {
	proxyToPython(w, r, "/simulate")
}

func handleScan(w http.ResponseWriter, r *http.Request) {
	proxyToPython(w, r, "/scan")
}

func handleCompareN(w http.ResponseWriter, r *http.Request) {
	proxyToPython(w, r, "/compare_n")
}

func main() {
	http.HandleFunc("/api/run", handleSimulate)
	http.HandleFunc("/api/scan", handleScan)
	http.HandleFunc("/api/compare_n", handleCompareN)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeCORS(w)
		if r.Method == "OPTIONS" {
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	fmt.Println("Go Orchestrator running on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}
