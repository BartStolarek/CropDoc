"use client";

import React from 'react';
import { Link } from "@nextui-org/link";
import { Snippet } from "@nextui-org/snippet";
import { Code } from "@nextui-org/code";
import { button as buttonStyles } from "@nextui-org/theme";

import { siteConfig } from "@/config/site";
import { title, subtitle } from "@/components/primitives";
import { GithubIcon } from "@/components/icons";
import { Navbar, NavbarBrand, NavbarContent, NavbarItem, NavbarMenuToggle, NavbarMenu, NavbarMenuItem } from "@nextui-org/react";
import CropDocLogo from "@/app/CropDocLogo.jsx";
import { Button } from "@nextui-org/button";
import { FileUpload } from "@/components/file-upload"; 


export default function Home() {
  
  const handleFileUpload = (files: File[]) => {
    console.log("Files uploaded:", files);
  };
  
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <h1 className="mb-4 text-3xl font-extrabold text-gray-900 dark:text-white md:text-5xl lg:text-6xl">
  <span className="">
    Welcome to&nbsp;
  </span>
  <span className="text-transparent bg-clip-text bg-gradient-to-r to-emerald-600 from-sky-400">
    CropDoc
  </span>
</h1>

      <p className="text-lg font-normal text-gray-500 lg:text-xl dark:text-gray-400">Computer vision model for crop disease detection.</p>
      <FileUpload onChange={handleFileUpload} /> {/* Add FileUpload component here */}

      <Button color="primary" isLoading>
        Click Me
      </Button>
    </div>
  );
}
