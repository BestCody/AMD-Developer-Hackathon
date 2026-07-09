import * as React from "react";

export interface BadgeProps {
  kind?: "processing" | "success" | "warning" | "error" | "neutral";
  children: React.ReactNode;
  style?: React.CSSProperties;
}

export function Badge(props: BadgeProps): JSX.Element;
