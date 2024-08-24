"use client";

import React, { useState } from 'react';
import { Button } from "@nextui-org/button";
import { FileUpload } from "@/components/file-upload"; 

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);

  const handleFileUpload = (files: File[]) => {
    console.log("Files uploaded:", files);
    if (files.length > 0) {
      setUploadedFile(files[0]); // Assuming single file upload
    }
  };

  const handlePredictClick = async () => {
    if (!uploadedFile) {
      console.error('No file uploaded');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      // Call the API on localhost:5000
      const response = await fetch('http://localhost:5000/pipeline/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      console.log('API response:', data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen space-y-32">
      <div className="flex flex-col items-center justify-center">
        <h1 className="mb-4 text-3xl font-extrabold text-gray-900 dark:text-white md:text-5xl lg:text-6xl">
          <span className="">
            Welcome to&nbsp;
          </span>
          <span className="text-transparent bg-clip-text bg-gradient-to-r to-emerald-600 from-sky-400">
            CropDoc
          </span>
        </h1>

        <p className="text-lg font-normal text-gray-500 lg:text-xl dark:text-gray-400">
          Computer vision model for crop disease detection.
        </p>
      </div>
      <FileUpload onChange={handleFileUpload} /> {/* Add FileUpload component here */}

      <Button color="primary" isLoading={isLoading} onClick={handlePredictClick}>
        Predict
      </Button>
    </div>
  );
}
