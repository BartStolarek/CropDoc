"use client";

import React, { useState } from "react";
import { Button } from "@nextui-org/button";
import { Card, CardHeader, CardBody, CardFooter } from "@nextui-org/card";

import { FileUpload } from "@/components/file-upload";
import config from "@/config/config";

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [predictionResults, setPredictionResults] = useState(null);

  const handleFileUpload = (files: File[]) => {
    /* eslint-disable no-console */
    console.log("Files uploaded:", files);
    if (files.length > 0) {
      setUploadedFile(files[0]); // Assuming single file upload
    }
  };

  const handlePredictClick = async () => {
    if (!uploadedFile) {
      /* eslint-disable no-console */
      console.error("No file uploaded");
      // eslint-disable-next-line padding-line-between-statements
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    // eslint-disable-next-line padding-line-between-statements
    formData.append("file", uploadedFile);

    try {
      // Call the API on localhost:5000
      const response = await fetch(`${config.apiUrl}/pipeline/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();

      console.log("API response:", data);
      setPredictionResults(data);
    } catch (error) {
      console.error("Error:", error);
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
