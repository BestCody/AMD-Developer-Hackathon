import * as React from "react";

export interface RadioProps {
  checked: boolean;
  onChange?: (checked: boolean) => void;
  label?: React.ReactNode;
  style?: React.CSSProperties;
}

export function Radio(props: RadioProps): JSX.Element;
