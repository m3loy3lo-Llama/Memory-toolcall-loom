import argparse
import sys
import torch
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from diffusers import AutoPipelineForText2Image

# Global pipeline reference
pipe = None

def get_torch_device_and_dtype(forced_device=None):
    if forced_device and forced_device != "auto":
        device = forced_device
        if device in ["mps", "cuda"]:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        print(f"💻 Local Diffusion: Forced device to {device} ({torch_dtype})", flush=True)
        return device, torch_dtype

    # Auto-detect check
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
        print("💻 Local Diffusion: Using auto-detected MPS (Apple Silicon GPU) acceleration", flush=True)
    elif torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        print("💻 Local Diffusion: Using auto-detected CUDA (Nvidia GPU) acceleration", flush=True)
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("💻 Local Diffusion: Using CPU fallback (Warning: this will be slow)", flush=True)
    return device, torch_dtype

def load_pipeline(forced_device=None):
    device, torch_dtype = get_torch_device_and_dtype(forced_device)
    try:
        print("⏳ Local Diffusion: Loading SD-Turbo model (this might take a few seconds, or minutes on first run)...", flush=True)
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch_dtype,
            variant="fp16" if torch_dtype == torch.float16 else None
        )
        pipeline.to(device)
        print("✅ Local Diffusion: Model loaded successfully!", flush=True)
        return pipeline
    except Exception as e:
        print(f"❌ Local Diffusion loading error: {str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)

def generate_image(pipeline, prompt, output_path):
    print(f"🎨 Local Diffusion: Generating image for prompt: '{prompt}'", flush=True)
    # For SD-Turbo, num_inference_steps=1 and guidance_scale=0.0 is standard and fast!
    image = pipeline(
        prompt=prompt,
        num_inference_steps=1,
        guidance_scale=0.0
    ).images[0]

    print(f"💾 Local Diffusion: Saving image to {output_path}", flush=True)
    image.save(output_path)
    print("✅ Local Diffusion: Generation completed successfully!", flush=True)

class LocalDiffusionHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Format logs to clean up standard library HTTP logging output
        sys.stdout.write(f"💻 Local Diffusion Server: {format % args}\n")
        sys.stdout.flush()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode("utf-8"))
                prompt = data.get("prompt")
                output_path = data.get("output")
                
                if not prompt or not output_path:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": "Missing prompt or output"}).encode("utf-8"))
                    return
                
                generate_image(pipe, prompt, output_path)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "output": output_path}).encode("utf-8"))
                
            except Exception as e:
                print(f"❌ Local Diffusion generation error: {str(e)}", file=sys.stderr, flush=True)
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port, forced_device=None):
    global pipe
    pipe = load_pipeline(forced_device)
    server_address = ('', port)
    httpd = HTTPServer(server_address, LocalDiffusionHandler)
    print(f"🚀 Local Diffusion Server ready on port {port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Local Diffusion Server...", flush=True)
        httpd.server_close()

def main():
    parser = argparse.ArgumentParser(description="Generate image using SD-Turbo locally.")
    parser.add_argument("--prompt", type=str, help="The prompt to generate the image.")
    parser.add_argument("--output", type=str, help="The path to save the generated image.")
    parser.add_argument("--server", action="store_true", help="Run as a persistent HTTP server.")
    parser.add_argument("--port", type=int, default=3002, help="The port to listen on in server mode.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="The torch device to use.")
    args = parser.parse_args()

    if args.server:
        run_server(args.port, args.device)
    else:
        if not args.prompt or not args.output:
            parser.error("either --server must be set, or both --prompt and --output must be provided.")
        pipeline = load_pipeline(args.device)
        generate_image(pipeline, args.prompt, args.output)

if __name__ == "__main__":
    main()
