package integration

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"context"

	"gpu-orchestrator/admission"
	"gpu-orchestrator/reconciler"
	"gpu-orchestrator/registry"
	schedulerpkg "gpu-orchestrator/scheduler"
)

func TestAdmissionGatewayReturns429WhenQueueFull(t *testing.T) {
	reg := registry.NewNodeRegistry()
	reg.Upsert(&registry.GPUNode{NodeID: "node-a", GPUID: "gpu-0", TotalVRAMMiB: 1000, UsedVRAMMiB: 1000}) // saturated

	packer := schedulerpkg.NewBinPacker(reg)
	rec := reconciler.NewReconciler(reg, make(chan reconciler.HeartbeatMsg, 1), make(chan reconciler.ReconcileAction, 8))
	gw := admission.NewGateway(packer, rec, admission.NewQueue(1), 200*time.Millisecond)

	srv := httptest.NewServer(http.HandlerFunc(gw.HandleSchedule))
	defer srv.Close()

	body := `{"ModelName":"model-z","VRAMNeededMiB":100,"Priority":2}` // P2 -> Low class

	go http.Post(srv.URL, "application/json", strings.NewReader(body)) // fills the one Low slot
	time.Sleep(20 * time.Millisecond)

	resp, err := http.Post(srv.URL, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("post failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("expected 429, got %d", resp.StatusCode)
	}
}

func TestAdmissionGatewayServesQueuedRequestOnceCapacityFrees(t *testing.T) {
	reg := registry.NewNodeRegistry()
	reg.Upsert(&registry.GPUNode{NodeID: "node-a", GPUID: "gpu-0", TotalVRAMMiB: 1000, UsedVRAMMiB: 1000})

	packer := schedulerpkg.NewBinPacker(reg)
	rec := reconciler.NewReconciler(reg, make(chan reconciler.HeartbeatMsg, 1), make(chan reconciler.ReconcileAction, 8))
	gw := admission.NewGateway(packer, rec, admission.NewQueue(10), 3*time.Second)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go gw.DrainLoop(ctx, 100*time.Millisecond)

	srv := httptest.NewServer(http.HandlerFunc(gw.HandleSchedule))
	defer srv.Close()

	go func() {
		time.Sleep(150 * time.Millisecond) // simulate a heartbeat reporting an eviction
		reg.Upsert(&registry.GPUNode{NodeID: "node-a", GPUID: "gpu-0", TotalVRAMMiB: 1000, UsedVRAMMiB: 100})
	}()

	body := `{"ModelName":"model-z","VRAMNeededMiB":200,"Priority":0}`
	resp, err := http.Post(srv.URL, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("post failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 once capacity freed, got %d", resp.StatusCode)
	}
}
