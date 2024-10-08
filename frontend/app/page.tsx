"use client";

import React, { useState } from "react";
import { Button } from "@nextui-org/button";
import { Card, CardHeader, CardBody, CardFooter } from "@nextui-org/card";

import { FileUpload } from "@/components/file-upload";
import config from "@/config/config";

// First, let's define interfaces for our prediction results
interface ClassProbabilities {
  [key: string]: number;
}

interface Prediction {
  prediction: string;
  probability: number;
  class_probabilities: ClassProbabilities;
}

interface PredictionResults {
  crop: Prediction;
  state: Prediction;
}

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [predictionResults, setPredictionResults] = 
  useState<PredictionResults | null>(null);


  const handleFileUpload = (files: File[]) => {
    /* eslint-disable no-console */
    console.log("Files uploaded:", files);
    if (files.length > 0) {
      setUploadedFile(files[0]); // Assuming single file upload
    } else {
      setUploadedFile(null); // Clear the file if no file is uploaded
    }
  };

  const handlePredictClick = async () => {
    if (!uploadedFile) {
      console.error("No file uploaded");
      return;
    }
  
    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", uploadedFile);
  
    const predict_api = `${process.env.NEXT_PUBLIC_API_URL}/pipeline/predict`;
    console.log("Attempting to call API at:", predict_api);
  
    try {
      console.log("API call started");
      const response = await fetch(predict_api, {
        method: "POST",
        body: formData,
      });
  
      console.log("API response status:", response.status);
      console.log("API response headers:", Object.fromEntries(response.headers.entries()));
  
      if (!response.ok) {
        const errorBody = await response.text();
        console.error("API error response:", errorBody);
        throw new Error(`Network response was not ok: ${response.status} ${response.statusText}`);
      }
  
      const data = await response.json();
      console.log("API response data:", data);
      setPredictionResults(data as PredictionResults);
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        console.error("Network error: Unable to reach the API. This might be due to CORS issues or the API being unreachable.");
      } else if (error instanceof Error) {
        console.error("Error during API call:", error.message);
      } else {
        console.error("An unknown error occurred:", error);
      }
      // Optionally set an error state here to display to the user
      // setError("An error occurred while processing your request. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="flex flex-col items-center justify-center min-h-screen space-y-32">
      <div className="flex flex-col items-center justify-center">
        <h1 className="mb-4 text-3xl font-extrabold text-gray-900 dark:text-white md:text-5xl lg:text-6xl">
          <span className="">Welcome to&nbsp;</span>
          <span className="text-transparent bg-clip-text bg-gradient-to-r to-emerald-600 from-sky-400">
            CropDoc
          </span>
        </h1>

        <p className="text-lg font-normal text-gray-500 lg:text-xl dark:text-gray-400">
          Computer vision model for crop disease detection.
        </p>
      </div>
      <FileUpload onChange={handleFileUpload} />

      <Button
        color="primary"
        isLoading={isLoading}
        onClick={handlePredictClick}
      >
        Predict
      </Button>

      <Card className="w-full max-w-3xl">
        <CardHeader>
          <h4 className="text-2xl font-bold">Prediction Results</h4>
        </CardHeader>
        <CardBody>
          {predictionResults ? (
            <div className="space-y-6">
              {/* Crop Prediction */}
              <div>
                <h5 className="text-xl font-semibold mb-2">Crop Prediction</h5>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-lg">
                    {predictionResults.crop.prediction}
                  </span>
                  <span className="bg-blue-500 text-white px-2 py-1 rounded-full text-sm">
                    {(predictionResults.crop.probability * 100).toFixed(2)}%
                  </span>
                </div>
                {Object.entries(predictionResults.crop.class_probabilities).map(
                  ([crop, prob]) => (
                    <div key={crop} className="mb-2">
                      <div className="flex justify-between mb-1">
                        <span>{crop}</span>
                        <span>{(prob * 100).toFixed(2)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div
                          className="bg-blue-600 h-2.5 rounded-full"
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                    </div>
                  ),
                )}
              </div>

              {/* State Prediction */}
              <div>
                <h5 className="text-xl font-semibold mb-2">State Prediction</h5>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-lg">
                    {predictionResults.state.prediction}
                  </span>
                  <span className="bg-green-500 text-white px-2 py-1 rounded-full text-sm">
                    {(predictionResults.state.probability * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="max-h-64 overflow-y-auto">
                  {Object.entries(predictionResults.state.class_probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([state, prob]) => (
                      <div key={state} className="mb-2">
                        <div className="flex justify-between mb-1">
                          <span>{state}</span>
                          <span>{(prob * 100).toFixed(2)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div
                            className="bg-green-600 h-2.5 rounded-full"
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          ) : (
            <p>
              No prediction results yet. Upload an image and click
              &quot;Predict&quot; to see results.
            </p>
          )}
        </CardBody>
        <CardFooter>
          <Button
            color="primary"
            isLoading={isLoading}
            onClick={handlePredictClick}
          >
            {predictionResults ? "Predict Again" : "Predict"}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
