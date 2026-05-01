import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.get("/", (req, res) => {
  res.send("RealityStream AI Assistant is running.");
});

app.post("/agent", async (req, res) => {
  try {
    const { message } = req.body;

    if (!message) {
      return res.status(400).json({ error: "Message is required" });
    }

    const model = genAI.getGenerativeModel({
      model: "gemini-2.0-flash",
    });

    const prompt = `
You are a RealityStream AI assistant.

User request:
${message}

If YAML is requested, return clean YAML only.
Otherwise give a clear short answer.
`;

    const result = await model.generateContent(prompt);
    const reply = result.response.text();

    res.json({ reply });

  } catch (error) {
    console.error("Agent error:", error);

    res.json({
      reply: "Fallback: YAML generation example\n\nfolder: demo\nmodels: rfc"
    });
  }
});

const port = process.env.PORT || 3000;

app.listen(port, () => {
  console.log(`RealityStream AI Assistant running on http://localhost:${port}`);
});