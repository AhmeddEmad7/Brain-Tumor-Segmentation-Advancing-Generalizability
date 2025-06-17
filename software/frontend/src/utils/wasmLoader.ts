export async function loadWasmModule(wasmPath: string) {
    try {
        const response = await fetch(wasmPath);
        const wasmBuffer = await response.arrayBuffer();
        const wasmModule = await WebAssembly.instantiate(wasmBuffer);
        return wasmModule.instance.exports;
    } catch (error) {
        console.error('Error loading WASM module:', error);
        throw error;
    }
} 