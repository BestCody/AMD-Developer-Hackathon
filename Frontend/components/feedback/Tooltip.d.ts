import * as React from "react";

export interface TooltipProps {
  label: React.ReactNode;
  children: React.ReactNode;
}

export function Tooltip(props: TooltipProps): JSX.Element;
